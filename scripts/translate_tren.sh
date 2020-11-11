#!/usr/bin/env bash

N=$1
M=$2
Device=$3

export CUDA_VISIBLE_DEVICES=$3



python3 ../translate.py \
    --model_name transformer_$2 \
    --source_path "/home/user_data/wengrx/WMT17-TREN/newstest$1.tc.bpe.tr" \
    --model_path "../save/$2/transformer_$2.best.tpz" \
    --config_path "../configs/transformer_$2_config.yaml" \
    --batch_size 50 \
    --beam_size 5 \
    --save_path "../result/$2/$2.newstest$1.txt" \
    --source_bpe_codes "" \
    --use_gpu

perl ../tools/multi-bleu.perl -lc /home/user_data/wengrx/WMT17-TREN/newstest$1.tc.en < ../result/$2/$2.newstest$1.txt.0