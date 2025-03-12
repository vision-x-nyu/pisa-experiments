import os
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from pprint import pformat

import torch
import torch.distributed as dist
import wandb
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.datasets.aspect import get_num_frames, get_image_size
from opensora.datasets.utils import save_sample
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import load, model_gathering, model_sharding, record_model_param_shape, save
from opensora.utils.config_utils import define_experiment_workspace, parse_configs, save_training_config
from opensora.utils.lr_scheduler import LinearWarmupLR
from opensora.utils.misc import (
    Timer,
    all_reduce_mean,
    create_logger,
    create_tensorboard_writer,
    format_numel_str,
    get_model_numel,
    requires_grad,
    to_torch_dtype,
)
from opensora.utils.train_utils import MaskGenerator, create_colossalai_plugin, update_ema
from opensora.utils.inference_utils import prepare_multi_resolution_info, collect_references_batch, apply_mask_strategy

import sys

sys.path = ["sam2"] + sys.path
from sam2.build_sam import build_sam2_video_predictor


def generate_masks(first_frame_mask, video, predictor):
    # video: [T, H, W, C]
    inference_state = predictor.init_state(video)
    predictor.reset_state(inference_state)
    for ann_bj_id in range(first_frame_mask.shape[0]):
        mask = first_frame_mask[ann_bj_id]
        predictor.add_new_mask(
            inference_state,
            frame_idx=0,
            obj_id=ann_bj_id,
            mask=mask,
        )
    video_segments = []
    for _, _, out_mask_logits in predictor.propagate_in_video(inference_state):
        frame_segment = out_mask_logits.squeeze().sigmoid()
        video_segments.append(frame_segment)
    video_segments = torch.stack(video_segments)
    return video_segments


def get_iou(masks1, masks2, eps=1e-6):
    # masks1: [T, N, H, W], masks2: [T, N, H, W] dtype: float
    intersection = (masks1 * masks2).sum(dim=(2, 3))
    union = masks1.sum(dim=(2, 3)) + masks2.sum(dim=(2, 3)) - intersection + eps
    iou = intersection / union  # [T, N]
    return iou


def compute_segloss(video_predictor, samples, gt_masks):
    # samples: [B, T, H, W, C], gt_masks: [B, T, N, H, W]
    num_samples = samples.shape[0]
    segloss_list = []
    for i in range(num_samples):
        sample = samples[i]  # [T, H, W, C]
        first_frame_mask = gt_masks[i, 0]  # [N, H, W]
        gt_segments = gt_masks[i]  # [T, N, H, W]
        pd_segments = generate_masks(first_frame_mask, sample, video_predictor)  # [T, N, H, W]
        pd_segments = pd_segments.reshape(gt_segments.shape)  # To avoid shape mismatch
        iou = get_iou(pd_segments, gt_segments)  # [T, N]
        segloss = 1 - iou
        segloss_list.append(segloss)
    segloss = torch.stack(segloss_list)  # [B, T, N]
    return torch.mean(segloss)


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=True)
    record_time = cfg.get("record_time", False)

    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    # == colossalai init distributed training ==
    # NOTE: A very large timeout is set to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(cfg.get("seed", 1024))
    coordinator = DistCoordinator()
    device = get_current_device()

    # == init exp_dir ==
    exp_name, exp_dir = define_experiment_workspace(cfg)
    coordinator.block_all()
    if coordinator.is_master():
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(cfg.to_dict(), exp_dir)
    coordinator.block_all()

    # == init logger, tensorboard & git  ==
    logger = create_logger(exp_dir)
    logger.info("Experiment directory created at %s", exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get("wandb", False):
            wandb.init(project="oro_exps", entity="nyu-visionx", name=exp_name, config=cfg.to_dict())
    dist.barrier()

    # == init ColossalAI booster ==
    plugin = create_colossalai_plugin(
        plugin=cfg.get("plugin", "zero2"),
        dtype=cfg_dtype,
        grad_clip=cfg.get("grad_clip", 0),
        sp_size=cfg.get("sp_size", 1),
        reduce_bucket_size_in_m=cfg.get("reduce_bucket_size_in_m", 20),
    )
    booster = Booster(plugin=plugin)
    torch.set_num_threads(1)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", None),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    num_steps_per_epoch = len(dataloader)

    if cfg.get("val_dataset", False):
        val_dataset = build_module(cfg.val_dataset, DATASETS)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.get("batch_size", None),
            num_workers=cfg.get("num_workers", 4),
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.get("text_encoder", None), MODELS, device=device, dtype=dtype)
    if text_encoder is not None:
        text_encoder_output_dim = text_encoder.output_dim
        text_encoder_model_max_length = text_encoder.model_max_length
    else:
        text_encoder_output_dim = cfg.get("text_encoder_output_dim", 4096)
        text_encoder_model_max_length = cfg.get("text_encoder_model_max_length", 300)

    # == build vae ==
    vae = build_module(cfg.get("vae", None), MODELS)
    if vae is not None:
        vae = vae.to(device, dtype).eval()
    if vae is not None:
        input_size = (dataset.num_frames, *dataset.image_size)
        latent_size = vae.get_latent_size(input_size)
        vae_out_channels = vae.out_channels
    else:
        latent_size = (None, None, None)
        vae_out_channels = cfg.get("vae_out_channels", 4)

    # == build diffusion model ==
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae_out_channels,
            caption_channels=text_encoder_output_dim,
            model_max_length=text_encoder_model_max_length,
            enable_sequence_parallelism=cfg.get("sp_size", 1) > 1,
        )
        .to(device, dtype)
        .train()
    )
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "[Diffusion] Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )

    # == build ema for diffusion model ==
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)
    ema.eval()
    update_ema(ema, model, decay=0, sharded=False)
    # == build sam2 model ==
    video_predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "models/sam2.1_hiera_large.pt",
        device=device,
    )
    for param in video_predictor.parameters():
        param.requires_grad = False
    # == setup loss function, build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # == setup optimizer ==
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        adamw_mode=True,
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("adam_eps", 1e-8),
    )

    warmup_steps = cfg.get("warmup_steps", None)

    if warmup_steps is None:
        lr_scheduler = None
    else:
        lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=cfg.get("warmup_steps"))

    # == additional preparation ==
    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)
    if cfg.get("mask_ratios", None) is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)

    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
    logger.info("Preparing for distributed training...")
    # == boosting ==
    # NOTE: we set dtype first to make initialization of model consistent with the dtype; then reset it to the fp32 as we make diffusion scheduler in fp32
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boosting model for distributed training")

    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)
    start_epoch = start_step = log_step = acc_step = 0
    running_loss = 0.0
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == resume ==
    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint")
        ret = load(
            booster,
            cfg.load,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            sampler=None if cfg.get("start_from_scratch", False) else sampler,
        )
        if not cfg.get("start_from_scratch", False):
            start_epoch, start_step = ret
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)

    model_sharding(ema)

    # =======================================================
    # 5. training loop
    # =======================================================
    dist.barrier()
    timers = {}
    timer_keys = [
        "move_data",
        "encode",
        "mask",
        "diffusion",
        "backward",
        "update_ema",
        "reduce_loss",
    ]
    for key in timer_keys:
        if record_time:
            timers[key] = Timer(key, coordinator=coordinator)
        else:
            timers[key] = nullcontext()
    # !!!
    text_encoder.y_embedder = model.module.y_embedder  # copied from inference.py to avoid error
    image_size = cfg.get("image_size", None)
    num_frames = get_num_frames(cfg.num_frames)
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    align = cfg.get("align", None)

    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)

        if cfg.get("val_dataset", False) and epoch % cfg.get("eval_every", 1) == 0:
            logger.info(f"Running inference evaluation at epoch {epoch}!")
            run_eval(model, vae, scheduler, text_encoder, val_dataloader, cfg, epoch, exp_dir)
            dist.barrier()

        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            for step, batch in pbar:
                timer_list = []
                batch_size = len(batch["video"])
                model_args = prepare_multi_resolution_info(
                    cfg.get("multi_resolution", None), batch_size, image_size, num_frames, dataset.fps, device, dtype
                )
                for i in range(len(batch["path"])):
                    batch["path"][i] = os.path.join(batch["path"][i], "rgba_00000.jpg")
                refs = collect_references_batch(batch["path"], vae, image_size)
                z = torch.randn(batch_size, vae.out_channels, *latent_size, device=device, dtype=dtype)
                masks = apply_mask_strategy(z, refs, ["0"] * batch_size, 0, align=align)
                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    prompts=batch["text"],
                    device=device,
                    additional_args=model_args,
                    progress=True,
                    mask=masks,
                )
                samples = vae.decode(samples.to(dtype), num_frames=num_frames)  # [B, C, T, H, W]
                samples = (
                    samples.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 4, 1)
                )  # [B, C, T, H, W] -> [B, T, H, W, C]
                # TODO: segloss
                gt_mask = batch.pop("seg").to(device)  # [B, T, N, H, W]
                loss = compute_segloss(video_predictor, samples, gt_mask)
                # == backward & update ==
                with timers["backward"] as backward_t:
                    booster.backward(loss=loss, optimizer=optimizer)
                    optimizer.step()
                    optimizer.zero_grad()

                    # update learning rate
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                if record_time:
                    timer_list.append(backward_t)

                # == update EMA ==
                with timers["update_ema"] as ema_t:
                    update_ema(ema, model.module, optimizer=optimizer, decay=cfg.get("ema_decay", 0.9999))
                if record_time:
                    timer_list.append(ema_t)

                # == update log info ==
                with timers["reduce_loss"] as reduce_loss_t:
                    all_reduce_mean(loss)
                    running_loss += loss.item()
                    global_step = epoch * num_steps_per_epoch + step
                    log_step += 1
                    acc_step += 1
                if record_time:
                    timer_list.append(reduce_loss_t)

                # == logging ==
                if coordinator.is_master() and (global_step + 1) % cfg.get("log_every", 1) == 0:
                    avg_loss = running_loss / log_step
                    # progress bar
                    pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                    # tensorboard
                    tb_writer.add_scalar("loss", loss.item(), global_step)
                    # wandb
                    if cfg.get("wandb", False):
                        wandb_dict = {
                            "iter": global_step,
                            "acc_step": acc_step,
                            "epoch": epoch,
                            "loss": loss.item(),
                            "avg_loss": avg_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                        if record_time:
                            wandb_dict.update(
                                {
                                    "debug/backward_time": backward_t.elapsed_time,
                                    "debug/update_ema_time": ema_t.elapsed_time,
                                    "debug/reduce_loss_time": reduce_loss_t.elapsed_time,
                                }
                            )
                        wandb.log(wandb_dict, step=global_step)

                    running_loss = 0.0
                    log_step = 0

                dist.barrier()

                # == checkpoint saving ==
                ckpt_every = cfg.get("ckpt_every", 0)
                if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                    model_gathering(ema, ema_shape_dict)
                    save_dir = save(
                        booster,
                        exp_dir,
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        sampler=sampler,
                        epoch=epoch,
                        step=step + 1,
                        global_step=global_step + 1,
                        batch_size=cfg.get("batch_size", None),
                    )
                    if dist.get_rank() == 0:
                        model_sharding(ema)
                    logger.info(
                        "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                        epoch,
                        step + 1,
                        global_step + 1,
                        save_dir,
                    )
                if record_time:
                    log_str = f"Rank {dist.get_rank()} | Epoch {epoch} | Step {step} | "
                    for timer in timer_list:
                        log_str += f"{timer.name}: {timer.elapsed_time:.3f}s | "
                    print(log_str)

        sampler.reset()
        start_step = 0


@torch.no_grad()
def run_eval(
    model,
    vae,
    scheduler,
    text_encoder,
    val_dataloader,
    cfg,
    epoch,
    exp_dir,
):
    if dist.get_rank() != 0:
        return
    save_dir = os.path.join(exp_dir, "samples", f"epoch_{epoch:03d}")
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    vae.eval()
    text_encoder.y_embedder = model.module.y_embedder  # copied from inference.py to avoid error
    image_size = cfg.get("image_size", None)
    num_frames = get_num_frames(cfg.num_frames)
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    device = "cuda"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    align = cfg.get("align", None)
    for _, batch in enumerate(iter(val_dataloader)):
        batch_size = len(batch["video"])
        model_args = prepare_multi_resolution_info(
            cfg.get("multi_resolution", None),
            batch_size,
            image_size,
            num_frames,
            val_dataloader.dataset.fps,
            device,
            dtype,
        )
        refs = collect_references_batch(batch["path"], vae, image_size)
        z = torch.randn(batch_size, vae.out_channels, *latent_size, device=device, dtype=dtype)
        masks = apply_mask_strategy(z, refs, ["0"] * batch_size, 0, align=align)
        samples = scheduler.sample(
            model,
            text_encoder,
            z=z,
            prompts=batch["text"],
            device=device,
            additional_args=model_args,
            progress=True,
            mask=masks,
        )
        samples = vae.decode(samples.to(dtype), num_frames=num_frames)
        for i, s in enumerate(samples):
            sample_dir = os.path.join(save_dir, batch["split"][i])
            os.makedirs(sample_dir, exist_ok=True)
            sample_num = batch["sample"][i]

            sample_save_path = os.path.join(sample_dir, f"{sample_num:03d}")
            save_sample(s, save_path=sample_save_path, fps=cfg.save_fps)

            gt_save_path = os.path.join(sample_dir, f"{sample_num:03d}_gt")
            save_sample(batch["video"][i], save_path=gt_save_path, fps=cfg.save_fps)

    model.train()
    vae.train()


if __name__ == "__main__":
    main()
