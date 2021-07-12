#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

wavpath_file="/data/lichunyou/aishell_data/kaldi_ark_format/test.txt"
model_file="model_out/epoch45_train:1.5883274685115967_val:8.618933424688823.pth.tar"
output_file="test_dataset/aishell_text_infer"
conf="conf/config.conf"
beam_size=3
is_cuda=True

python infer.py                      \
    --recog-conf ${conf}             \
    --model-file ${model_file}       \
    --wav-path-file ${wavpath_file}  \
    --output-txt ${output_file}       \
    --beam-size ${beam_size}         \
    --nbest ${beam_size}             
