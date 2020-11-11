#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

N=$1
MODEL=$2
export MODEL_NAME="transformer"

python -m src.bin.translate \
    --model_name $MODEL_NAME \
    --source_path "/home/user_data/weihr/NMT_DATA_PY3/WMT14-DE-EN/newstest$1.tok.de" \
    --reference_path "/home/user_data/weihr/NMT_DATA_PY3/WMT14-DE-EN/newstest$1.tok.en" \
    --model_path "./save/$2/$MODEL_NAME.best.final" \
    --config_path "./configs/transformer_$2.yaml" \
    --batch_size 20 \
    --beam_size 4 \
    --saveto "./result/$2/$MODEL_NAME.newstest$1.txt" \
    --use_gpu