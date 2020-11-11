#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

N=$1
MODEL=$2
export MODEL_NAME="transformer"

mkdir result_$2

python -m src.bin.translate \
    --model_name $MODEL_NAME \
    --source_path "/home/user_data/weihr/NMT_DATA_PY3/NIST-ZH-EN/test/mt$1.src" \
    --model_path "./save_$2_nist_zhen/$MODEL_NAME.best.final" \
    --config_path "./configs/transformer_nist_$2_zhen.yaml" \
    --batch_size 20 \
    --beam_size 5 \
    --saveto "./result_$2_nist_zhen/$MODEL_NAME.mt$1.txt" \
    --use_gpu


perl /home/wengrx/Tools/multi-bleu.perl /home/user_data/weihr/NMT_DATA_PY3/NIST-ZH-EN/test/mt$1.ref < ./result_$2_nist_zhen/$MODEL_NAME.mt$1.txt.0