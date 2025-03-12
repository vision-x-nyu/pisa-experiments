export CUDA_VISIBLE_DEVICES=0,1,
export WANDB_DIR=/data/pisa/wandb
torchrun --standalone --nproc_per_node 2 scripts/oro_flow.py \
    configs/post_training/oro_flow.py \
    --data-path configs/dataset/oro.csv \
    --ckpt-path /data/pisa/checkpoints/base
