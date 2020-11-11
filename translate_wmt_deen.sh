#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

N=$1
MODEL=$2
export MODEL_NAME="transformer"
mkdir -p ./result/$MODEL_NAME-$2
python -m src.bin.translate \
    --model_name $MODEL_NAME \
    --source_path "/home/user_data/wengrx/WMT14-DE-EN-clean/newstest$1.tok.de" \
    --reference_path "/home/user_data/wengrx/WMT14-DE-EN-clean/newstest$1.en" \
    --model_path "./save/$MODEL_NAME-$2/$MODEL_NAME.best.final" \
    --config_path "./configs/$MODEL_NAME-$2.yaml" \
    --batch_size 30 \
    --beam_size 4 \
    --saveto "./result/$MODEL_NAME-$2/$MODEL_NAME.newstest$1.txt" \
    --use_gpu