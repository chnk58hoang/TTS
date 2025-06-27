#! /bin/bash


dataset_path=/data/tts/female
meta_file_train=metadata.csv
formatter=ns_female
batch_size=12
epochs=1500
continue_path=
output_path=/TTS/recipes/vivoice/naturalspeech/models/"${formatter}"
test_dir=/data/tts/101_speakers
checkpoint_path=
dict_phonemes_json=/TTS/TTS/tts/utils/text/vietnamese/dict_phoneme.json
gpu=0
accum=1

python train_openvoice.py \
    --dataset_path "${dataset_path}" \
    --meta_file_train "${meta_file_train}" \
    --formatter "${formatter}" \
    --batch_size "${batch_size}" \
    --epochs "${epochs}" \
    --restore_path "${continue_path}" \
    --output_path "${output_path}" \
    --test_dir "${test_dir}" \
    --checkpoint_path "${checkpoint_path}" \
    --dict_phonemes_json "${dict_phonemes_json}" \
    --gpu "${gpu}" \
    --accum "${accum}"


