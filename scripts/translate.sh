#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

N=$1
M=$2

python3 ../translate.py \
    --model_name transformer_$2 \
    --source_path "/home/user_data/wengrx/NIST_1.34M/test/MT0$1/src.txt" \
    --model_path "../save/$2/transformer_$2.best.tpz" \
    --config_path "../configs/transformer_$2_config.yaml" \
    --batch_size 20 \
    --beam_size 5 \
    --save_path "../result/$2/$2.MT0$1.txt" \
    --source_bpe_codes "" \
    --use_gpu

perl ../tools/multi-bleu.perl /home/user_data/wengrx/NIST_1.34M/test/MT0$1/en. < ../result/$2/$2.MT0$1.txt.0