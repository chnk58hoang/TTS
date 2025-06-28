#! /bin/bash


dataset_path=/data/vivoice/
meta_file_train=transcripts.txt
formatter=vivoice
batch_size=24
epochs=100
continue_path=
output_path=/TTS/recipes/vie_vc/openvoice/models/"${formatter}"
test_dir=/data/tts/101_speaker/wavs
checkpoint_path=/TTS/checkpoints/checkpoint.pth
dict_phonemes_json=/TTS/TTS/tts/utils/text/vietnamese/dict_phoneme.json
gpu=0
accum=1
sr=16000

python train_openvoice.py \
    --dataset_path "${dataset_path}" \
    --meta_file_train "${meta_file_train}" \
    --formatter "${formatter}" \
    --batch_size "${batch_size}" \
    --epochs "${epochs}" \
    --sr "${sr}" \
    --restore_path "${continue_path}" \
    --output_path "${output_path}" \
    --test_dir "${test_dir}" \
    --pretrain_path "${checkpoint_path}" \
    --dict_phonemes_json "${dict_phonemes_json}" \
    --gpu "${gpu}" \
    --accum "${accum}"


