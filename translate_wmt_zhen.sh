#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

N=$1
MODEL=$2
GPU=$3
export MODEL_NAME="transformer"

mkdir -p result/$2

python -m src.bin.translate \
    --model_name $MODEL_NAME \
    --source_path "/home/user_data55/wengrx/nmt_data/WMT17-ZHEN/no_un/newstest$1.tok.zh" \
    --reference_path "/home/user_data55/wengrx/nmt_data/WMT17-ZHEN/no_un/newstest$1.tok.en" \
    --model_path "/home/user_data55/wengrx/nmt_model/$2/$MODEL_NAME.best.final" \
    --config_path "./configs/transformer-$2.yaml" \
    --batch_size 20 \
    --beam_size 4 \
    --saveto "./result/$2/$MODEL_NAME.newstest$1.txt" \
    --use_gpu