export CUDA_VISIBLE_DEVICES=0,1,
export WANDB_DIR=/data/pisa/wandb
torchrun --standalone --nproc_per_node 2 scripts/oro_seg.py \
    configs/post_training/oro_seg.py \
    --data-path configs/dataset/oro.csv \
    --ckpt-path /data/pisa/checkpoints/base
