# Data and eval settings

multi_resolution = "STDiT2"
image_size = (256, 256)
num_frames = 32
align = 5
save_fps = 16

dataset = dict(
    type="VideoTextMaskDataset",
    data_path="configs/dataset/oro.csv",
    transform_name="resize_crop",
    num_frames=num_frames,
    image_size=image_size,
    extra_keys=True,
    N=1000,
    fps=16,
)
val_dataset = dict(
    type="VideoTextDataset",
    data_path="configs/dataset/val.csv",
    transform_name="resize_crop",
    num_frames=num_frames,
    image_size=image_size,
    extra_keys=True,
    fps=16.0,
)


grad_checkpoint = True

# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
plugin = "zero2"

# Model settings
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False,
    freeze_y_embedder=True,
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
    # shardformer=True,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
)

mask_ratios = {"image_head": 1.0}

# Log settings
seed = 42
wandb = True
epochs = 1000
log_every = 10
ckpt_every = 500
eval_every = 2

# optimization settings
load = None
grad_clip = 1.0
lr = 1e-6
ema_decay = 0.99
adam_eps = 1e-15
warmup_steps = 0
batch_size = 1
outputs = "/data/pisa/output/depth"