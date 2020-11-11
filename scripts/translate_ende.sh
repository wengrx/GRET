#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

N=$1
Model_Name=$2
Specific_Name=$3

python3 ../translate.py \
    --model_name transformer_$2 \
    --source_path "/home/user_data/wengrx/WMT16_ENDE/newstest$1.tok.bpe.32000.en" \
    --model_path "../save/$2_$3/transformer_$2.best.tpz" \
    --config_path "../configs/transformer_$2_config.yaml" \
    --specific_path "../configs/$3.yaml" \
    --batch_size 50 \
    --beam_size 5 \
    --save_path "../result/$2_$3/$2_$3.newstest$1.txt" \
    --source_bpe_codes "" \
    --use_gpu

perl ../tools/multi-bleu.perl -lc /home/user_data/wengrx/WMT16_ENDE/newstest$1.tok.de < ../result/$2_$3/$2_$3.newstest$1.txt.0