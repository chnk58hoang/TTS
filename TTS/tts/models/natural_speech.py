import math
import os
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from coqpit import Coqpit
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from trainer.torch import DistributedSampler, DistributedSamplerWrapper
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.datasets.dataset import TTSDataset, _parse_sample
from TTS.tts.layers.vits.discriminator import VitsDiscriminator
from TTS.tts.layers.naturalspeech.networks import (DurationPredictor, LearnableUpsampling,
                                                   TextEncoder, ResidualCouplingBlocks,
                                                   PosteriorEncoder, VAEMemoryBank)
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.fairseq import rehash_fairseq_vits_checkpoint
from TTS.tts.utils.helpers import generate_path, maximum_path, rand_segments, segment, sequence_mask
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.characters import BaseCharacters, BaseVocabulary, _characters, _pad, _phonemes, _punctuations
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment
from TTS.utils.io import load_fsspec
from TTS.utils.samplers import BucketBatchSampler
from TTS.vocoder.models.hifigan_generator import HifiganGenerator
from TTS.vocoder.utils.generic_utils import plot_results


hann_window = {}
mel_basis = {}


@torch.no_grad()
def weights_reset(m: nn.Module):
    # check if the current module has reset_parameters and if it is reset the weight
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def get_module_weights_sum(mdl: nn.Module):
    dict_sums = {}
    for name, w in mdl.named_parameters():
        if "weight" in name:
            value = w.data.sum().item()
            dict_sums[name] = value
    return dict_sums


def load_audio(file_path):
    """Load the audio file normalized in [-1, 1]

    Return Shapes:
        - x: :math:`[1, T]`
    """
    x, sr = torchaudio.load(file_path)
    assert (x > 1).sum() + (x < -1).sum() == 0
    return x, sr


def _amp_to_db(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def _db_to_amp(x, C=1):
    return torch.exp(x) / C


def amp_to_db(magnitudes):
    output = _amp_to_db(magnitudes)
    return output


def db_to_amp(magnitudes):
    output = _db_to_amp(magnitudes)
    return output


def wav_to_spec(y, n_fft, hop_length, win_length, center=False):
    """
    Args Shapes:
        - y : :math:`[B, 1, T]`

    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    y = y.squeeze(1)

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_length) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel(spec, n_fft, num_mels, sample_rate, fmin, fmax):
    """
    Args Shapes:
        - spec : :math:`[B,C,T]`

    Return Shapes:
        - mel : :math:`[B,C,T]`
    """
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    mel = torch.matmul(mel_basis[fmax_dtype_device], spec)
    mel = amp_to_db(mel)
    return mel


def wav_to_mel(y, n_fft, num_mels, sample_rate, hop_length, win_length, fmin, fmax, center=False):
    """
    Args Shapes:
        - y : :math:`[B, 1, T]`

    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    y = y.squeeze(1)

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_length) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = amp_to_db(spec)
    return spec


#############################
# CONFIGS
#############################


@dataclass
class NaturalSpeechAudioConfig(Coqpit):
    fft_size: int = 1024
    sample_rate: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    num_mels: int = 80
    mel_fmin: int = 0
    mel_fmax: int = None


##############################
# DATASET
##############################


def get_attribute_balancer_weights(items: list, attr_name: str, multi_dict: dict = None):
    """Create inverse frequency weights for balancing the dataset.
    Use `multi_dict` to scale relative weights."""
    attr_names_samples = np.array([item[attr_name] for item in items])
    unique_attr_names = np.unique(attr_names_samples).tolist()
    attr_idx = [unique_attr_names.index(l) for l in attr_names_samples]
    attr_count = np.array([len(np.where(attr_names_samples == l)[0]) for l in unique_attr_names])
    weight_attr = 1.0 / attr_count
    dataset_samples_weight = np.array([weight_attr[l] for l in attr_idx])
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    if multi_dict is not None:
        # check if all keys are in the multi_dict
        for k in multi_dict:
            assert k in unique_attr_names, f"{k} not in {unique_attr_names}"
        # scale weights
        multiplier_samples = np.array([multi_dict.get(item[attr_name], 1.0) for item in items])
        dataset_samples_weight *= multiplier_samples
    return (
        torch.from_numpy(dataset_samples_weight).float(),
        unique_attr_names,
        np.unique(dataset_samples_weight).tolist(),
    )


class NaturalSpeechDataset(TTSDataset):
    def __init__(self, model_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_id = self.tokenizer.characters.pad_id
        self.model_args = model_args

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_text = item["text"]

        wav, _ = load_audio(item["audio_file"])
        if self.model_args.encoder_sample_rate is not None:
            if wav.size(1) % self.model_args.encoder_sample_rate != 0:
                wav = wav[:, : -int(wav.size(1) % self.model_args.encoder_sample_rate)]

        wav_filename = os.path.basename(item["audio_file"])

        token_ids = self.get_token_ids(idx, item["text"])

        # after phonemization the text length may change
        # this is a shameful 🤭 hack to prevent longer phonemes
        # TODO: find a better fix
        if len(token_ids) > self.max_text_len or wav.shape[1] < self.min_audio_len:
            self.rescue_item_idx += 1
            return self.__getitem__(self.rescue_item_idx)

        return {
            "raw_text": raw_text,
            "token_ids": token_ids,
            "token_len": len(token_ids),
            "wav": wav,
            "wav_file": wav_filename,
            "speaker_name": item["speaker_name"],
            "language_name": item["language"],
            "audio_unique_name": item["audio_unique_name"],
        }

    @property
    def lengths(self):
        lens = []
        for item in self.samples:
            _, wav_file, *_ = _parse_sample(item)
            audio_len = os.path.getsize(wav_file) / 16 * 8  # assuming 16bit audio
            lens.append(audio_len)
        return lens

    def collate_fn(self, batch):
        """
        Return Shapes:
            - tokens: :math:`[B, T]`
            - token_lens :math:`[B]`
            - token_rel_lens :math:`[B]`
            - waveform: :math:`[B, 1, T]`
            - waveform_lens: :math:`[B]`
            - waveform_rel_lens: :math:`[B]`
            - speaker_names: :math:`[B]`
            - language_names: :math:`[B]`
            - audiofile_paths: :math:`[B]`
            - raw_texts: :math:`[B]`
            - audio_unique_names: :math:`[B]`
        """
        # convert list of dicts to dict of lists
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x.size(1) for x in batch["wav"]]), dim=0, descending=True
        )

        max_text_len = max([len(x) for x in batch["token_ids"]])
        token_lens = torch.LongTensor(batch["token_len"])
        token_rel_lens = token_lens / token_lens.max()

        wav_lens = [w.shape[1] for w in batch["wav"]]
        wav_lens = torch.LongTensor(wav_lens)
        wav_lens_max = torch.max(wav_lens)
        wav_rel_lens = wav_lens / wav_lens_max

        token_padded = torch.LongTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, wav_lens_max)
        token_padded = token_padded.zero_() + self.pad_id
        wav_padded = wav_padded.zero_() + self.pad_id
        for i in range(len(ids_sorted_decreasing)):
            token_ids = batch["token_ids"][i]
            token_padded[i, : batch["token_len"][i]] = torch.LongTensor(token_ids)

            wav = batch["wav"][i]
            wav_padded[i, :, : wav.size(1)] = torch.FloatTensor(wav)

        return {
            "tokens": token_padded,
            "token_lens": token_lens,
            "token_rel_lens": token_rel_lens,
            "waveform": wav_padded,  # (B x T)
            "waveform_lens": wav_lens,  # (B)
            "waveform_rel_lens": wav_rel_lens,
            "speaker_names": batch["speaker_name"],
            "language_names": batch["language_name"],
            "audio_files": batch["wav_file"],
            "raw_text": batch["raw_text"],
            "audio_unique_names": batch["audio_unique_name"],
        }


class NaturalSpeech(BaseTTS):
    def __init__(self,
                 config: Coqpit,
                 ap: "AudioProcessor" = None,
                 tokenizer: TTSTokenizer = None,
                 speaker_manager: SpeakerManager = None,
                 language_manager: LanguageManager = None,):
        super().__init__(config, ap, tokenizer, speaker_manager, language_manager)
        self.init_multispeaker(config)
        self.init_multilingual(config)
        self.init_upsampling(config)

        self.text_encoder = TextEncoder(n_vocab=self.args.num_chars,
                                        out_channels=self.args.hidden_channels,
                                        hidden_channels=self.args.hidden_channels,
                                        hidden_channels_ffn=self.args.hidden_channels_ffn_text_encoder,
                                        num_heads=self.args.num_heads_text_encoder,
                                        num_layers=self.args.num_layers_text_encoder,
                                        kernel_size=self.args.kernel_size_text_encoder,
                                        dropout=self.args.dropout_p_text_encoder,
                                        language_emb_dim=self.embedded_speaker_dim)

        self.posterior_encoder = PosteriorEncoder(in_channels=self.args.out_channels,
                                                  out_channels=self.args.hidden_channels,
                                                  hidden_channels=self.args.hidden_channels,
                                                  kernel_size=self.args.kernel_size_posterior_encoder,
                                                  dilation_rate=self.args.dilation_rate_posterior_encoder,
                                                  num_layers=self.args.num_layers_posterior_encoder,
                                                  cond_channels=self.embedded_speaker_dim)

        self.duration_predictor = DurationPredictor(in_channels=self.args.hidden_channels,
                                                    filter_channels=self.args.hidden_channels,
                                                    kernel_size=self.args.dp_kernel_size,
                                                    p_dropout=self.args.dropout_dp,
                                                    gin_channels=self.embedded_speaker_dim)

        self.learnable_upsampling = LearnableUpsampling(d_predictor=self.args.d_predictor,
                                                        kernel_size=self.args.lu_kernel_size,
                                                        dropout=self.args.dropout_lu,
                                                        conv_output_size=self.args.lu_conv_output_size,
                                                        dim_w=self.args.lu_dim_w,
                                                        dim_c=self.args.lu_dim_c,
                                                        max_seq_len=self.lu_max_seq_len)

        self.flow = ResidualCouplingBlocks(
            channels=self.args.hidden_channels,
            hidden_channels=self.args.hidden_channels,
            kernel_size=self.args.kernel_size_flow,
            dilation_rate=self.args.dilation_rate_flow,
            num_layers=self.args.num_layers_flow,
            cond_channels=self.embedded_speaker_dim,
        )

        self.waveform_decoder = HifiganGenerator(
            self.args.hidden_channels,
            1,
            self.args.resblock_type_decoder,
            self.args.resblock_dilation_sizes_decoder,
            self.args.resblock_kernel_sizes_decoder,
            self.args.upsample_kernel_sizes_decoder,
            self.args.upsample_initial_channel_decoder,
            self.args.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

        if self.args.init_discriminator:
            self.disc = VitsDiscriminator(
                periods=self.args.periods_multi_period_discriminator,
                use_spectral_norm=self.args.use_spectral_norm_disriminator,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def init_multispeaker(self, config: Coqpit):
        """Initialize multi-speaker modules of a model. A model can be trained either with a speaker embedding layer
        or with external `d_vectors` computed from a speaker encoder model.

        You must provide a `speaker_manager` at initialization to set up the multi-speaker modules.

        Args:
            config (Coqpit): Model configuration.
            data (List, optional): Dataset items to infer number of speakers. Defaults to None.
        """
        self.embedded_speaker_dim = 0
        self.num_speakers = self.args.num_speakers
        self.audio_transform = None

        if self.speaker_manager:
            self.num_speakers = self.speaker_manager.num_speakers

        if self.args.use_speaker_embedding:
            self._init_speaker_embedding()

        if self.args.use_d_vector_file:
            self._init_d_vector()

        # TODO: make this a function
        if self.args.use_speaker_encoder_as_loss:
            if self.speaker_manager.encoder is None and (
                not self.args.speaker_encoder_model_path or not self.args.speaker_encoder_config_path
            ):
                raise RuntimeError(
                    " [!] To use the speaker consistency loss (SCL) you need to specify speaker_encoder_model_path and speaker_encoder_config_path !!"
                )

            self.speaker_manager.encoder.eval()
            print(" > External Speaker Encoder Loaded !!")

            if (
                hasattr(self.speaker_manager.encoder, "audio_config")
                and self.config.audio.sample_rate != self.speaker_manager.encoder.audio_config["sample_rate"]
            ):
                self.audio_transform = torchaudio.transforms.Resample(
                    orig_freq=self.config.audio.sample_rate,
                    new_freq=self.speaker_manager.encoder.audio_config["sample_rate"],
                )

    def _init_speaker_embedding(self):
        # pylint: disable=attribute-defined-outside-init
        if self.num_speakers > 0:
            print(" > initialization of speaker-embedding layers.")
            self.embedded_speaker_dim = self.args.speaker_embedding_channels
            self.emb_g = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)

    def _init_d_vector(self):
        # pylint: disable=attribute-defined-outside-init
        if hasattr(self, "emb_g"):
            raise ValueError("[!] Speaker embedding layer already initialized before d_vector settings.")
        self.embedded_speaker_dim = self.args.d_vector_dim

    def init_multilingual(self, config: Coqpit):
        """Initialize multilingual modules of a model.

        Args:
            config (Coqpit): Model configuration.
        """
        if self.args.language_ids_file is not None:
            self.language_manager = LanguageManager(language_ids_file_path=config.language_ids_file)

        if self.args.use_language_embedding and self.language_manager:
            print(" > initialization of language-embedding layers.")
            self.num_languages = self.language_manager.num_languages
            self.embedded_language_dim = self.args.embedded_language_dim
            self.emb_l = nn.Embedding(self.num_languages, self.embedded_language_dim)
            torch.nn.init.xavier_uniform_(self.emb_l.weight)
        else:
            self.embedded_language_dim = 0

    def init_upsampling(self):
        """
        Initialize upsampling modules of a model.
        """
        if self.args.encoder_sample_rate:
            self.interpolate_factor = self.config.audio["sample_rate"] / self.args.encoder_sample_rate
            self.audio_resampler = torchaudio.transforms.Resample(
                orig_freq=self.config.audio["sample_rate"], new_freq=self.args.encoder_sample_rate
            )  # pylint: disable=W0201

    def on_epoch_start(self, trainer):  # pylint: disable=W0613
        """Freeze layers at the beginning of an epoch"""
        self._freeze_layers()
        # set the device of speaker encoder
        if self.args.use_speaker_encoder_as_loss:
            self.speaker_manager.encoder = self.speaker_manager.encoder.to(self.device)

    def on_init_end(self, trainer):  # pylint: disable=W0613
        """Reinit layes if needed"""
        if self.args.reinit_DP:
            before_dict = get_module_weights_sum(self.duration_predictor)
            # Applies weights_reset recursively to every submodule of the duration predictor
            self.duration_predictor.apply(fn=weights_reset)
            after_dict = get_module_weights_sum(self.duration_predictor)
            for key, value in after_dict.items():
                if value == before_dict[key]:
                    raise RuntimeError(" [!] The weights of Duration Predictor was not reinit check it !")
            print(" > Duration Predictor was reinit.")

        if self.args.reinit_text_encoder:
            before_dict = get_module_weights_sum(self.text_encoder)
            # Applies weights_reset recursively to every submodule of the duration predictor
            self.text_encoder.apply(fn=weights_reset)
            after_dict = get_module_weights_sum(self.text_encoder)
            for key, value in after_dict.items():
                if value == before_dict[key]:
                    raise RuntimeError(" [!] The weights of Text Encoder was not reinit check it !")
            print(" > Text Encoder was reinit.")

    def get_aux_input(self, aux_input: Dict):
        sid, g, lid, _ = self._set_cond_input(aux_input)
        return {"speaker_ids": sid, "style_wav": None, "d_vectors": g, "language_ids": lid}

    def _freeze_layers(self):
        if self.args.freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            if hasattr(self, "emb_l"):
                for param in self.emb_l.parameters():
                    param.requires_grad = False

        if self.args.freeze_PE:
            for param in self.posterior_encoder.parameters():
                param.requires_grad = False

        if self.args.freeze_DP:
            for param in self.duration_predictor.parameters():
                param.requires_grad = False

        if self.args.freeze_flow_decoder:
            for param in self.flow.parameters():
                param.requires_grad = False

        if self.args.freeze_waveform_decoder:
            for param in self.waveform_decoder.parameters():
                param.requires_grad = False

    @staticmethod
    def _set_cond_input(aux_input: Dict):
        """Set the speaker conditioning input based on the multi-speaker mode."""
        sid, g, lid, durations = None, None, None, None
        if "speaker_ids" in aux_input and aux_input["speaker_ids"] is not None:
            sid = aux_input["speaker_ids"]
            if sid.ndim == 0:
                sid = sid.unsqueeze_(0)
        if "d_vectors" in aux_input and aux_input["d_vectors"] is not None:
            g = F.normalize(aux_input["d_vectors"]).unsqueeze(-1)
            if g.ndim == 2:
                g = g.unsqueeze_(0)

        if "language_ids" in aux_input and aux_input["language_ids"] is not None:
            lid = aux_input["language_ids"]
            if lid.ndim == 0:
                lid = lid.unsqueeze_(0)

        if "durations" in aux_input and aux_input["durations"] is not None:
            durations = aux_input["durations"]

        return sid, g, lid, durations

    def _set_speaker_input(self, aux_input: Dict):
        d_vectors = aux_input.get("d_vectors", None)
        speaker_ids = aux_input.get("speaker_ids", None)

        if d_vectors is not None and speaker_ids is not None:
            raise ValueError("[!] Cannot use d-vectors and speaker-ids together.")

        if speaker_ids is not None and not hasattr(self, "emb_g"):
            raise ValueError("[!] Cannot use speaker-ids without enabling speaker embedding.")

        g = speaker_ids if speaker_ids is not None else d_vectors
        return g

    def forward_mas(self, outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g, lang_emb):
        # find the alignment path
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        with torch.no_grad():
            o_scale = torch.exp(-2 * logs_p)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.einsum("klm, kln -> kmn", [o_scale, -0.5 * (z_p**2)])
            logp3 = torch.einsum("klm, kln -> kmn", [m_p * o_scale, z_p])
            logp4 = torch.sum(-0.5 * (m_p**2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp2 + logp3 + logp1 + logp4
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t, t']

        # duration predictor
        attn_durations = attn.sum(3)
        if self.args.use_sdp:
            loss_duration = self.duration_predictor(
                x.detach() if self.args.detach_dp_input else x,
                x_mask,
                attn_durations,
                g=g.detach() if self.args.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.args.detach_dp_input and lang_emb is not None else lang_emb,
            )
            loss_duration = loss_duration / torch.sum(x_mask)
        else:
            attn_log_durations = torch.log(attn_durations + 1e-6) * x_mask
            log_durations = self.duration_predictor(
                x.detach() if self.args.detach_dp_input else x,
                x_mask,
                g=g.detach() if self.args.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.args.detach_dp_input and lang_emb is not None else lang_emb,
            )
            loss_duration = torch.sum((log_durations - attn_log_durations) ** 2, [1, 2]) / torch.sum(x_mask)
        outputs["loss_duration"] = loss_duration
        return outputs, attn
    