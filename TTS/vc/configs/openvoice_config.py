from dataclasses import dataclass, field
from typing import List, Dict
from coqpit import Coqpit
from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.vc.models.openvoice import OVArgs, OVAudioConfig


@dataclass
class OpenVoiceConfig(BaseTTSConfig):
    """OpenVoice configuration

    Args:
        spec_channels (int):
            The number of channels in the spectrogram.
        use_speaker_embedding (bool):
            Whether to use speaker embedding.
        use_language_embedding (bool):
            Whether to use language embedding.
        use_text_embedding (bool):
            Whether to use text embedding.
    """
    model: str = "OpenVoice"
    model_args: OVArgs = field(default_factory=OVArgs)
    audio: OVAudioConfig = field(default_factory=OVAudioConfig)
    text_cleaner: str = "english_cleaners"
    use_phonemes: bool = False
    phoneme_language: str = "vi"
    phoneme_cache_path: str = "phoneme_cache"
    compute_input_seq_cache: bool = False

    # optimizer
    grad_clip: List[float] = field(default_factory=lambda: [1000, 1000])
    lr_gen: float = 0.0002
    lr_disc: float = 0.0002
    lr_scheduler_gen: str = "ExponentialLR"
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    lr_scheduler_disc: str = "ExponentialLR"
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    scheduler_after_epoch: bool = True
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01})
    # loss params
    kl_loss_alpha: float = 1.0
    disc_loss_alpha: float = 1.0
    gen_loss_alpha: float = 1.0
    feat_loss_alpha: float = 1.0
    mel_loss_alpha: float = 45.0

    # data loader params
    return_wav: bool = True
    compute_linear_spec: bool = True

    # sampler params
    use_weighted_sampler: bool = False  # TODO: move it to the base config
    weighted_sampler_attrs: dict = field(default_factory=lambda: {})
    weighted_sampler_multipliers: dict = field(default_factory=lambda: {})

    # overrides
    r: int = 1  # DO NOT CHANGE
    add_blank: bool = True

    # testing
    test_samples: List[Dict] = field(default_factory=lambda: [{}])

    def __post_init__(self):
        for key, val in self.model_args.items():
            if hasattr(self, key):
                self[key] = val
