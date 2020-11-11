#!/usr/bin/env bash

Model=$1

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train \
    --model_name "transformer" \
    --reload \
    --config_path "./configs/transformer_$1.yaml" \
    --log_path "./log_$1" \
    --saveto "./save_$1/" \
    --use_gpu