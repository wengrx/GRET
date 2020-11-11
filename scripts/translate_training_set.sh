#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2


#export THEANO_FLAGS=device=cuda3,floatX=float32,mode=FAST_RUN
export MODEL_NAME="transformer_base"

python3 ../translate.py \
    --model_name $MODEL_NAME \
    --source_path "/home/user_data/weihr/NMT_DATA_PY3/1.34M/train/zh.under50.txt" \
    --model_path "../save/base/$MODEL_NAME.best.tpz" \
    --config_path "../configs/transformer_base_config.yaml" \
    --batch_size 20 \
    --beam_size 5 \
    --saveto "../result/base/en.under50.trans.txt" \
    --source_bpe_codes "" \
    --use_gpu
