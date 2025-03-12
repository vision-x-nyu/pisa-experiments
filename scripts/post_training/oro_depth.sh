export CUDA_VISIBLE_DEVICES=0,1,
export WANDB_DIR=/data/pisa/wandb
torchrun --standalone --nproc_per_node 2 scripts/oro_depth.py \
    configs/post_training/oro_depth.py \
    --data-path configs/dataset/oro.csv \
    --ckpt-path /data/pisa/checkpoints/base
