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
from TTS.tts.layers.vits.networks import TextEncoder
from TTS.vc.models.base_vc import BaseVC
from TTS.vc.models.freevc import Generator, ResidualCouplingBlock
from TTS.tts.utils.fairseq import rehash_fairseq_vits_checkpoint
from TTS.tts.utils.helpers import generate_path, maximum_path, rand_segments, segment, sequence_mask
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.characters import BaseCharacters, BaseVocabulary, _characters, _pad, _phonemes, _punctuations
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment
from TTS.utils.io import load_fsspec
from TTS.utils.samplers import BucketBatchSampler
from TTS.vocoder.utils.generic_utils import plot_results

from TTS.vc.modules.freevc import commons as commons
from TTS.vc.modules.freevc import modules as modules
from torch.nn.utils.parametrizations import weight_norm


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


def _amp_to_db(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def amp_to_db(magnitudes):
    output = _amp_to_db(magnitudes)
    return output


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


class PosteriorEncoder(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None, tau=1.0):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * tau * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels, gin_channels=0, layernorm=True):
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, gin_channels)
        if layernorm:
            self.layernorm = nn.LayerNorm(self.spec_channels)
        else:
            self.layernorm = None

    def forward(self, inputs, mask=None):
        N = inputs.size(0)

        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        if self.layernorm is not None:
            out = self.layernorm(out)

        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0))

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


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


class OpenVoice(BaseVC):
    """
    Synthesizer for Training
    """

    def __init__(self, config: Coqpit, speaker_manager: SpeakerManager = None):
        super().__init__(config, None, speaker_manager, None)
        self.init_multispeaker(config)

        self.spec_segment_size = self.args.spec_segment_size
        self.spec_channels = self.args.spec_channels
        self.inter_channels = self.args.inter_channels
        self.hidden_channels = self.args.hidden_channels
        self.hidden_channels_ffn = self.args.hidden_channels_ffn
        self.filter_channels = self.args.filter_channels
        self.n_heads = self.args.n_heads
        self.n_layers = self.args.n_layers
        self.kernel_size = self.args.kernel_size
        self.p_dropout = self.args.p_dropout
        self.resblock = self.args.resblock
        self.resblock_kernel_sizes = self.args.resblock_kernel_sizes
        self.resblock_dilation_sizes = self.args.resblock_dilation_sizes
        self.upsample_rates = self.args.upsample_rates
        self.upsample_initial_channel = self.args.upsample_initial_channel
        self.upsample_kernel_sizes = self.args.upsample_kernel_sizes
        self.gin_channels = self.args.gin_channels
        self.ssl_dim = self.args.ssl_dim
        self.use_spk = self.args.use_spk
        self.n_vocab = self.args.n_vocab

        self.phone_encoder = TextEncoder(n_vocab=self.args.n_vocab,
                                         out_channels=self.inter_channels,
                                         hidden_channels=self.hidden_channels,
                                         hidden_channels_ffn=self.hidden_channels_ffn,
                                         num_heads=self.n_heads,
                                         num_layers=self.n_layers,
                                         kernel_size=self.kernel_size,
                                         dropout_p=self.p_dropout,
                                         language_emb_dim=self.gin_channels)

        self.dec = Generator(initial_channel=self.inter_channels,
                             resblock=self.resblock,
                             resblock_kernel_sizes=self.resblock_kernel_sizes,
                             resblock_dilation_sizes=self.resblock_dilation_sizes,
                             upsample_rates=self.upsample_rates,
                             upsample_initial_channel=self.upsample_initial_channel,
                             upsample_kernel_sizes=self.upsample_kernel_sizes,
                             gin_channels=self.gin_channels)
        self.enc_q = PosteriorEncoder(
            self.spec_channels,
            self.inter_channels,
            self.hidden_channels,
            5,
            1,
            16,
            gin_channels=self.gin_channels,
        )

        self.flow = ResidualCouplingBlock(self.inter_channels, self.hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        self.ref_enc = ReferenceEncoder(self.spec_channels, self.gin_channels)
        self.disc = VitsDiscriminator(periods=self.args.periods_multi_period_discriminator,
                                      use_spectral_norm=self.args.use_spectral_norm_disriminator,)

    def forward_mas(self, z_p, m_p, logs_p, x_mask, y_mask):
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)  # [b, 1, t_text, t_spec]
        with torch.no_grad():
            o_scale = torch.exp(-2 * logs_p)  # [b, d, t_text]
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]).unsqueeze(-1)  # [b, t_text, 1]
            logp2 = torch.einsum("klm, kln -> kmn", [o_scale, -0.5 * (z_p**2)])  # [b, t_text, t_spec]
            logp3 = torch.einsum("klm, kln -> kmn", [m_p * o_scale, z_p])  # [b, t_text, t_spec]
            logp4 = torch.sum(-0.5 * (m_p**2) * o_scale, [1]).unsqueeze(-1)  # [b, t_text, 1]
            logp = logp2 + logp3 + logp1 + logp4  # [b, t_text, t_spec]
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t_text, t_spec]
        return attn

    def forward(self, x, x_lengths, y, y_lengths, gt_waveform):
        src = self.ref_enc(y, mask=None)
        x, m_p, logs_p, x_mask = self.phone_encoder(x, x_lengths, lang_emb=None)
        z_q, m_q, logs_q, y_mask = self.enc_q(y, y_lengths)
        z_slice_q, q_slice_idx = rand_segments(z_q, y_lengths, self.spec_segment_size)
        gt_waveform_seg = segment(gt_waveform,
                                  q_slice_idx * self.config.audio_hop_length,
                                  self.spec_segment_size * self.config.audio_hop_length,
                                  pad_short=True)
        output_wave = self.dec(z_slice_q)
        z_p = self.flow(z_q, y_mask, g=src)
        attn = self.forward_mas(z_p, m_p, logs_p, x_mask, y_mask)
        m_p_aligned = torch.einsum("klmn, kjm -> kjn", [attn, m_p])  # [batch, 1, t_text, t_spec] x [batch, d, t_text] -> [batch, d, t_spec]
        logs_p_aligned = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])  # [batch, 1, t_text, t_spec] x [batch, d, t_text] -> [batch, d, t_spec]
        z_p_aligned = m_p_aligned + torch.randn_like(m_p_aligned) * torch.exp(logs_p_aligned)
        output = {
            "gt_wave_seg": gt_waveform_seg,
            "out_wave_seg": output_wave,
            "q_slice_idx": q_slice_idx,
            "z_p": z_p,
            "logs_q": logs_q,
            "m_p_aligned": m_p_aligned,
            "logs_p_aligned": logs_p_aligned,
        }
        return output

    @torch.no_grad()
    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt, tau=1.0):
        g_src = sid_src
        g_tgt = sid_tgt
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src if not self.zero_g else torch.zeros_like(g_src),
                                            tau=tau)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt if not self.zero_g else torch.zeros_like(g_tgt))
        return o_hat, y_mask, (z, z_p, z_hat)

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
        self.num_spks = self.args.num_spks
        if self.speaker_manager:
            self.num_spks = self.speaker_manager.num_spks

    @staticmethod
    def init_from_config(config):
        model = OpenVoice(config, speaker_manager=None)
        return model

    def load_checkpoint(self, config, checkpoint_path, eval=False, strict=True, cache=False):
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        self.load_state_dict(state["model"], strict=strict)
        if eval:
            self.eval()

    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        spec_lens = batch["spec_lens"]
        if optimizer_idx == 0:
            tokens = batch["tokens"]
            token_lengths = batch["token_lens"]
            spec = batch["spec"]
            waveform = batch["waveform"]

            outputs = self.forward(tokens, token_lengths, spec, spec_lens, waveform)
            self.model_outputs_cache = outputs
            scores_disc_fake, _, scores_disc_real, _ = self.disc(
                outputs["out_wave_seg"].detach(), outputs['gt_wave_seg']
            )
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    scores_disc_real,
                    scores_disc_fake,
                )
            return outputs, loss_dict
        if optimizer_idx == 1:
            mel = batch["mel"]

            with autocast(enabled=False):
                mel_slice = segment(mel.float(),
                                    self.model_outputs_cache["q_slice_idx"],
                                    self.spec_segment_size,
                                    pad_short=True)
                mel_slice_hat = wav_to_mel(
                    y=self.model_outputs_cache["out_wave_seg"].float(),
                    n_fft=self.config.audio.fft_size,
                    sample_rate=self.config.audio.sample_rate,
                    num_mels=self.config.audio.num_mels,
                    hop_length=self.config.audio.hop_length,
                    win_length=self.config.audio.win_length,
                    fmin=self.config.audio.mel_fmin,
                    fmax=self.config.audio.mel_fmax,
                    center=False,
                )
            scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.disc(
                self.model_outputs_cache["out_wave_seg"], self.model_outputs_cache["gt_wave_seg"]
            )
            # compute losses
            with autocast(enabled=False):
                pass

    def get_criterion(self):
        from TTS.tts.layers.losses import OVGeneratorLoss, OVDiscriminatorLoss
        return [OVDiscriminatorLoss(self.config), OVGeneratorLoss(self.config)]

    def format_batch(self, batch: Dict) -> Dict:
        """Compute speaker, langugage IDs and d_vector for the batch if necessary."""
        speaker_ids = None
        language_ids = None
        d_vectors = None

        # get numerical speaker ids from speaker names
        if self.speaker_manager is not None and self.speaker_manager.name_to_id and self.args.use_speaker_embedding:
            speaker_ids = [self.speaker_manager.name_to_id[sn] for sn in batch["speaker_names"]]

        if speaker_ids is not None:
            speaker_ids = torch.LongTensor(speaker_ids)

        # get d_vectors from audio file names
        if self.speaker_manager is not None and self.speaker_manager.embeddings and self.args.use_d_vector_file:
            d_vector_mapping = self.speaker_manager.embeddings
            d_vectors = [d_vector_mapping[w]["embedding"] for w in batch["audio_unique_names"]]
            d_vectors = torch.FloatTensor(d_vectors)

        # get language ids from language names
        if self.language_manager is not None and self.language_manager.name_to_id and self.args.use_language_embedding:
            language_ids = [self.language_manager.name_to_id[ln] for ln in batch["language_names"]]

        if language_ids is not None:
            language_ids = torch.LongTensor(language_ids)

        batch["language_ids"] = language_ids
        batch["d_vectors"] = d_vectors
        batch["speaker_ids"] = speaker_ids
        return batch

    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device."""
        ac = self.config.audio

        if self.args.encoder_sample_rate:
            wav = self.audio_resampler(batch["waveform"])
        else:
            wav = batch["waveform"]

        # compute spectrograms
        batch["spec"] = wav_to_spec(wav, ac.fft_size, ac.hop_length, ac.win_length, center=False)

        if self.args.encoder_sample_rate:
            # recompute spec with high sampling rate to the loss
            spec_mel = wav_to_spec(batch["waveform"], ac.fft_size, ac.hop_length, ac.win_length, center=False)
            # remove extra stft frames if needed
            if spec_mel.size(2) > int(batch["spec"].size(2) * self.interpolate_factor):
                spec_mel = spec_mel[:, :, : int(batch["spec"].size(2) * self.interpolate_factor)]
            else:
                batch["spec"] = batch["spec"][:, :, : int(spec_mel.size(2) / self.interpolate_factor)]
        else:
            spec_mel = batch["spec"]

        batch["mel"] = spec_to_mel(
            spec=spec_mel,
            n_fft=ac.fft_size,
            num_mels=ac.num_mels,
            sample_rate=ac.sample_rate,
            fmin=ac.mel_fmin,
            fmax=ac.mel_fmax,
        )

        if self.args.encoder_sample_rate:
            assert batch["spec"].shape[2] == int(
                batch["mel"].shape[2] / self.interpolate_factor
            ), f"{batch['spec'].shape[2]}, {batch['mel'].shape[2]}"
        else:
            assert batch["spec"].shape[2] == batch["mel"].shape[2], f"{batch['spec'].shape[2]}, {batch['mel'].shape[2]}"

        # compute spectrogram frame lengths
        batch["spec_lens"] = (batch["spec"].shape[2] * batch["waveform_rel_lens"]).int()
        batch["mel_lens"] = (batch["mel"].shape[2] * batch["waveform_rel_lens"]).int()

        if self.args.encoder_sample_rate:
            assert (batch["spec_lens"] - (batch["mel_lens"] / self.interpolate_factor).int()).sum() == 0
        else:
            assert (batch["spec_lens"] - batch["mel_lens"]).sum() == 0

        # zero the padding frames
        batch["spec"] = batch["spec"] * sequence_mask(batch["spec_lens"]).unsqueeze(1)
        batch["mel"] = batch["mel"] * sequence_mask(batch["mel_lens"]).unsqueeze(1)
        return batch

    def get_sampler(self, config: Coqpit, dataset: TTSDataset, num_gpus=1, is_eval=False):
        weights = None
        data_items = dataset.samples
        if getattr(config, "use_weighted_sampler", False):
            for attr_name, alpha in config.weighted_sampler_attrs.items():
                print(f" > Using weighted sampler for attribute '{attr_name}' with alpha '{alpha}'")
                multi_dict = config.weighted_sampler_multipliers.get(attr_name, None)
                print(multi_dict)
                weights, attr_names, attr_weights = get_attribute_balancer_weights(
                    attr_name=attr_name, items=data_items, multi_dict=multi_dict
                )
                weights = weights * alpha
                print(f" > Attribute weights for '{attr_names}' \n | > {attr_weights}")

        # input_audio_lenghts = [os.path.getsize(x["audio_file"]) for x in data_items]

        if weights is not None:
            w_sampler = WeightedRandomSampler(weights, len(weights))
            batch_sampler = BucketBatchSampler(
                w_sampler,
                data=data_items,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                sort_key=lambda x: os.path.getsize(x["audio_file"]),
                drop_last=True,
            )
        else:
            batch_sampler = None
        # sampler for DDP
        if batch_sampler is None:
            batch_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        else:  # If a sampler is already defined use this sampler and DDP sampler together
            batch_sampler = (
                DistributedSamplerWrapper(batch_sampler) if num_gpus > 1 else batch_sampler
            )  # TODO: check batch_sampler with multi-gpu
        return batch_sampler

    def get_scheduler(self, optimizer) -> List:
        """Set the schedulers for each optimizer.

        Args:
            optimizer (List[`torch.optim.Optimizer`]): List of optimizers.

        Returns:
            List: Schedulers, one for each optimizer.
        """
        scheduler_D = get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer[0])
        scheduler_G = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[1])
        return [scheduler_D, scheduler_G]

    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters.
        It returnes 2 optimizers in a list. First one is for the generator and the second one is for the discriminator.
        Returns:
            List: optimizers.
        """
        # select generator parameters
        optimizer0 = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, self.disc)

        gen_parameters = chain(params for k, params in self.named_parameters() if not k.startswith("disc."))
        optimizer1 = get_optimizer(
            self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, parameters=gen_parameters
        )
        return [optimizer0, optimizer1]

    def get_lr(self) -> List:
        """Set the initial learning rates for each optimizer.

        Returns:
            List: learning rates for each optimizer.
        """
        return [self.config.lr_disc, self.config.lr_gen]
    
    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":
        if is_eval and not config.run_eval:
            loader = None
        else:
            # init dataloader
            dataset = VitsDataset(
                model_args=self.args,
                samples=samples,
                batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                verbose=verbose,
                tokenizer=self.tokenizer,
                start_by_longest=config.start_by_longest,
            )

            # wait all the DDP process to be ready
            if num_gpus > 1:
                dist.barrier()

            # sort input sequences from short to long
            dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(config, dataset, num_gpus)
            if sampler is None:
                loader = DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    shuffle=False,  # shuffle is done in the dataset.
                    collate_fn=dataset.collate_fn,
                    drop_last=False,  # setting this False might cause issues in AMP training.
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
            else:
                if num_gpus > 1:
                    loader = DataLoader(
                        dataset,
                        sampler=sampler,
                        batch_size=config.eval_batch_size if is_eval else config.batch_size,
                        collate_fn=dataset.collate_fn,
                        num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                        pin_memory=False,
                    )
                else:
                    loader = DataLoader(
                        dataset,
                        batch_sampler=sampler,
                        collate_fn=dataset.collate_fn,
                        num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                        pin_memory=False,
                    )
        return loader
