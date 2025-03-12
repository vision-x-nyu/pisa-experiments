export CUDA_VISIBLE_DEVICES=0

python scripts/inference.py \
    configs/inference/sample.py \
    --ckpt-path /data/pisa/checkpoints/base \
    --num-frames 32 \
    --fps 16 \
    --save-fps 16 \
    --image-size 256 256 \
    --save-dir /data/pisa/results/base \
    --prompt-path configs/inference/prompt.txt

python process_results.py \
    --root_path /data/pisa/results/base
