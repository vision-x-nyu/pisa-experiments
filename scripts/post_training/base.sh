export CUDA_VISIBLE_DEVICES=0,1,
export WANDB_DIR=/data/pisa/wandb
torchrun --standalone --nproc_per_node 2 scripts/train.py \
    configs/post_training/base.py \
    --data-path configs/dataset/psft.csv
