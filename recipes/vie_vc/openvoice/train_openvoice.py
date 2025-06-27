import os
import random
import argparse
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.vc.configs.openvoice_config import OpenVoiceConfig
from TTS.tts.datasets import load_tts_samples
from TTS.vc.models.openvoice import (OpenVoice,
                                     OVAudioConfig,
                                     OVCharacters)
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


def pick_multiple_audio_pairs(parent_dir, num_pairs):
    """
    Tạo `num_pairs` cặp audio từ các folder con của `parent_dir`,
    mỗi cặp là {'src': ..., 'tgt': ...} với src và tgt đến từ 2 folder khác nhau.
    """
    all_folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]

    if len(all_folders) < 2:
        raise ValueError("Cần ít nhất 2 thư mục con để chọn cặp audio.")

    def random_file_from(folder):
        folder_path = os.path.join(parent_dir, folder)
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        if not files:
            raise ValueError(f"Folder '{folder}' không chứa file nào.")
        return os.path.join(folder_path, random.choice(files))

    pairs = []
    for _ in range(num_pairs):
        folder1, folder2 = random.sample(all_folders, 2)  # khác folder
        src = random_file_from(folder1)
        tgt = random_file_from(folder2)
        pairs.append({"src": src, "tgt": tgt})

    return pairs


def main(args):
    openvoice_characters = OVCharacters()
    # dataset config
    dataset_config = BaseDatasetConfig(formatter=args.formatter,
                                       meta_file_train=args.meta_file_train,
                                       language='vi',
                                       path=args.dataset_path)
    # Audio config
    audio_config = OVAudioConfig(sample_rate=22050,
                                 win_length=1024,
                                 hop_length=256,
                                 num_mels=80,
                                 mel_fmin=0,
                                 mel_fmax=None)
    # test samples
    test_samples = pick_multiple_audio_pairs(args.test_dir, 5) if args.test_dir else []
    model_config = OpenVoiceConfig(
        audio=audio_config,
        run_name="openvoice_vietnamese",
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr_disc=args.lr,
        lr_gen=args.lr,
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
        compute_input_seq_cache=False,
        print_step=args.print_step,
        print_eval=False,
        mixed_precision=False,
        max_text_len=325,  # change this if you have a larger VRAM than 16GB
        datasets=[dataset_config],
        characters=openvoice_characters.to_config(),
        test_samples=test_samples
    )
    # Tokenizer is used to convert text to sequences of token IDs.
    tokenizer, model_config = TTSTokenizer.init_from_config(model_config)
    # Load samples from the dataset
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=model_config.eval_split_max_size,
        eval_split_size=model_config.eval_split_size)

    audio_processor = AudioProcessor.init_from_config(model_config)
    model = OpenVoice(model_config, audio_processor, tokenizer, speaker_manager=None)
    if args.pretrain_path:
        print(f"Loading pre-trained model from {args.pretrain_path}")
        model.load_checkpoint(config=None, checkpoint_path=args.pretrain_path, strict=False)
    trainer = Trainer(
        TrainerArgs(continue_path=args.continue_path,
                    restore_path=args.restore_path,
                    gpu=args.gpu,
                    grad_accum_steps=args.accum),
        model_config,
        args.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples)
    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--meta_file_train", type=str, required=True,
                        default='metadata.txt', help="Path to the meta file for training")
    parser.add_argument("--test_dir", type=str, required=False, default=None)
    parser.add_argument("--formatter", type=str, default="ns_female", help="dataset formatter")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for training")
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
    parser.add_argument("--pretrain_path", type=str, default=None,
                        help="Path to the pretrain model")
    parser.add_argument("--dict_phonemes_json", type=str, required=True,
                        help="Path to the phoneme dictionary JSON file",
                        default='TTS/TTS/tts/utils/text/vietnamese/dict_phoneme.json')
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use for training")
    parser.add_argument("--finetune", action='store_true', help="finetune or train from scratch")
    parser.add_argument("--multi_spk", action='store_true')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.output_path, exist_ok=True)
    main(args)
    print("Training completed successfully.")
    print("Model saved to:", args.output_path)
