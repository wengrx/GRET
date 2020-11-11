#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

N=$1
MODEL=$2
export MODEL_NAME="transformer"
mkdir -p ./result/$2
python -m src.bin.translate \
    --model_name $MODEL_NAME \
    --source_path "/home/user_data/wengrx/WMT14-DE-EN-clean/newstest$1.tok.en" \
    --reference_path "/home/user_data/wengrx/WMT14-DE-EN-clean/newstest$1.de" \
    --model_path "./save/$2/$MODEL_NAME.best.final" \
    --config_path "./configs/transformer_$2.yaml" \
    --batch_size 30 \
    --beam_size 4 \
    --saveto "./result/$2/$MODEL_NAME.newstest$1.txt" \
    --use_gpu