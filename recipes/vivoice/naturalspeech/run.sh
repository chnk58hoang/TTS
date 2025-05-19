#! /bin/bash


dataset_path=
meta_file_train=
formatter=
batch_size=
epochs=
output_path=
dict_phonemes_json=
gpu=0

python train_naturalspeech.py \
    --dataset_path "${dataset_path}" \
    --meta_file_train "${meta_file_train}" \
    --formatter "${formatter}" \
    --batch_size "${batch_size}" \
    --epochs "${epochs}" \
    --output_path "${output_path}" \
    --dict_phonemes_json "${dict_phonemes_json}" \
    --gpu "${gpu}"


