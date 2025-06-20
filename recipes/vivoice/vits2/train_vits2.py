import os
import argparse
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits2_config import Vits2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits2 import (Vits2,
                                  Vits2AudioConfig,
                                  Vits2Args,
                                  Vits2Characters)
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


def build_characters():
    vits2_charaters = Vits2Characters()
    return vits2_charaters


def get_configs(args):
    # dataset config
    dataset_config = BaseDatasetConfig(formatter=args.formatter,
                                       meta_file_train=args.meta_file_train,
                                       language='vi',
                                       path=args.dataset_path)

    # Audio config
    audio_config = Vits2AudioConfig(sample_rate=16000,
                                    win_length=1024,
                                    hop_length=256,
                                    num_mels=80,
                                    mel_fmin=0,
                                    mel_fmax=None)

    # model args
    vits2_args = Vits2Args()
    # model config
    model_config = Vits2Config(
        model_args=vits2_args,
        audio=audio_config,
        run_name="vits2_vietnamese",
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        batch_group_size=args.batch_group_size,
        num_loader_workers=args.num_loader_workers,
        num_eval_loader_workers=args.num_eval_loader_workers,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=args.epochs,
        text_cleaner="english_cleaners",
        use_phonemes=False,
        phoneme_language="vi",
        phoneme_cache_path=os.path.join(args.output_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        print_step=args.print_step,
        print_eval=False,
        mixed_precision=False,
        max_text_len=325,  # change this if you have a larger VRAM than 16GB
    )
    # print(model_config.kl_loss_fwd_alpha)
    return dataset_config, model_config


def build_tokenizer(model_config, characters):
    # Tokenizer is used to convert text to sequences of token IDs.
    # config is updated with the default characters if not defined in the config.
    # phomizer = ViPhonemizer()
    tokenizer = TTSTokenizer(use_phonemes=model_config.use_phonemes,
                             characters=characters,
                             phonemizer=None,
                             use_eos_bos=False)
    return tokenizer


def get_train_val_samples(dataset_config,
                          model_config):
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=model_config.eval_split_max_size,
        eval_split_size=model_config.eval_split_size)
    return train_samples, eval_samples


def main(args):
    dataset_config, model_config = get_configs(args)
    audio_processor = AudioProcessor.init_from_config(model_config)
    characters = build_characters()
    tokenizer = build_tokenizer(model_config, characters)
    train_samples, eval_samples = get_train_val_samples(dataset_config, model_config)
    if args.multi_spk:
        speaker_manager = SpeakerManager()
        speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
        model_config.model_args.num_speakers = speaker_manager.num_speakers
    else:
        speaker_manager = None
    ns_model = Vits2(model_config, audio_processor, tokenizer,
                     speaker_manager=speaker_manager)
    trainer = Trainer(
        TrainerArgs(continue_path=args.continue_path,
                    restore_path=args.restore_path,
                    gpu=args.gpu),
        model_config,
        args.output_path,
        model=ns_model,
        train_samples=train_samples,
        eval_samples=eval_samples)
    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--meta_file_train", type=str, required=True,
                        default='metadata.txt', help="Path to the meta file for training")
    parser.add_argument("--formatter", type=str, default="ns_female", help="dataset formatter")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--batch_group_size", type=int, default=16,
                        help="Batch group size for training")
    parser.add_argument("--num_loader_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--num_eval_loader_workers", type=int, default=4,
                        help="Number of workers for evaluation data loading")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train")
    parser.add_argument("--print_step", type=int, default=100,
                        help="Number of steps to print training progress")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the trained model and logs")
    parser.add_argument("--continue_path", type=str, required=False,
                        default="")
    parser.add_argument("--restore_path", type=str, default=None,
                        help="Path to restore the model from a checkpoint")
    parser.add_argument("--dict_phonemes_json", type=str, required=True,
                        help="Path to the phoneme dictionary JSON file",
                        default='TTS/TTS/tts/utils/text/vietnamese/dict_phoneme.json')
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use for training")
    parser.add_argument("--multi_spk", action='store_true')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.output_path, exist_ok=True)
    main(args)
    print("Training completed successfully.")
    print("Model saved to:", args.output_path)

