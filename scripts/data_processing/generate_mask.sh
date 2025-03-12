export CUDA_VISIBLE_DEVICES=0

python data_processing/generate_mask.py \
    --config_file configs/data_processing/mask_example.jsonl
    