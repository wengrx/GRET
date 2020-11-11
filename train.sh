#!/usr/bin/env bash
Model=$1
GPU=$2
export MODEL_NAME="transformer"
export CUDA_VISIBLE_DEVICES=$2
echo "Using GPU $CUDA_VISIBLE_DEVICES..."
echo "Training $MODEL_NAME-$1..."

python -m src.bin.train \
    --model_name $MODEL_NAME \
    --config_path "./configs/$MODEL_NAME-$1.yaml" \
    --log_path "./log/$MODEL_NAME-$1" \
    --saveto "./save/$MODEL_NAME-$1" \
    --valid_path "./valid/$MODEL_NAME-$1" \
    --use_gpu \
    --reload