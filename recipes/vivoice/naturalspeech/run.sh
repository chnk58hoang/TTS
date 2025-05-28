#! /bin/bash


dataset_path=/data/female
meta_file_train=metadata.csv
formatter=ns_female
batch_size=32
epochs=1500
continue_path=
output_path=/TTS/recipes/vivoice/naturalspeech/models/"${formatter}"
dict_phonemes_json=/TTS/TTS/tts/utils/text/vietnamese/dict_phoneme.json
gpu=0
multi_spk=True
warm_up=True

python train_naturalspeech.py \
    --dataset_path "${dataset_path}" \
    --meta_file_train "${meta_file_train}" \
    --formatter "${formatter}" \
    --batch_size "${batch_size}" \
    --epochs "${epochs}" \
    --output_path "${output_path}" \
    --dict_phonemes_json "${dict_phonemes_json}" \
    --gpu "${gpu}" \
    --multi_spk "${multi_spk}" \
    --warm_up "${warm_up}"


