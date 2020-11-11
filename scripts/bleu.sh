#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

N=$1
export MODEL_NAME="transformer"

perl ../tools/multi-bleu.perl /home/user_data/wengrx/NIST_1.34M/test/MT0$1/en. < ../result_base/$MODEL_NAME.MT0$1.txt.0