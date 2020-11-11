#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
Model_Name=$1
Specific_Name=$2
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python ../train.py \
    --model_name "transformer_$1" \
    --reload \
    --use_gpu \
    --config_path "../configs/transformer_$1_config.yaml" \
    --specific_path "../configs/$2.yaml" \
    --log_path "../log/$1_$2" \
    --save_path "../save/$1_$2"