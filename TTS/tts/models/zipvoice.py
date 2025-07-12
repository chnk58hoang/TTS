from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from coqpit import Coqpit
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.layers.zipvoice.zipformer import TTSZipformer
from TTS.tts.layers.zipvoice.scaling import ScheduledFloat
from TTS.tts.layers.zipvoice.utils import (AttributeDict,
                                           condition_time_mask,
                                           get_tokens_index,
                                           make_pad_mask,
                                           pad_labels,
                                           prepare_avg_tokens_durations,
                                           to_int_tuple)
from typing import List, Optional
from torch import nn
import torch


class ZipVoice(BaseTTS):
    def __init__(self,
                 config: Coqpit,
                 ap: "AudioProcessor" = None,
                 tokenizer: TTSTokenizer = None,
                 speaker_manager: SpeakerManager = None,
                 language_manager: LanguageManager = None):
        super().__init__(config, ap, tokenizer, speaker_manager, language_manager)
        self.text_encoder = TTSZipformer(in_dim=self.args.text_embed_dim,
                                         out_dim=self.args.feat_dim,
                                         downsampling_factor=self.args.text_encoder_downsampling_factor,
                                         num_encoder_layers=self.args.text_encoder_num_layers,
                                         cnn_module_kernel=self.args.text_encoder_cnn_module_kernel,
                                         encoder_dim=self.args.text_encoder_dim,
                                         feedforward_dim=self.args.text_encoder_feedforward_dim,
                                         num_heads=self.args.text_encoder_num_heads,
                                         query_head_dim=self.args.query_head_dim,
                                         pos_head_dim=self.args.pos_head_dim,
                                         value_head_dim=self.args.value_head_dim,
                                         pos_dim=self.args.pos_dim,
                                         dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
                                         warmup_batches=4000,
                                         use_time_embed=False)
        self.fm_decoder = TTSZipformer(in_dim=self.args.feat_dim * 3,
                                       out_dim=self.args.feat_dim,
                                       downsampling_factor=self.args.fm_decoder_downsampling_factor,
                                       num_encoder_layers=self.args.fm_decoder_num_layers,
                                       cnn_module_kernel=self.args.fm_decoder_cnn_module_kernel,
                                       encoder_dim=self.args.fm_decoder_dim,
                                       feedforward_dim=self.args.fm_decoder_feedforward_dim,
                                       num_heads=self.args.fm_decoder_num_heads,
                                       query_head_dim=self.args.query_head_dim,
                                       pos_head_dim=self.args.pos_head_dim,
                                       value_head_dim=self.args.value_head_dim,
                                       pos_dim=self.args.pos_dim,
                                       dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
                                       warmup_batches=4000.0,
                                       use_time_embed=True,
                                       time_embed_dim=192,
                                       use_guidance_scale_embed=False)
        self.embed = nn.Embedding(self.args.num_chars,
                                  self.args.text_embed_dim)

    def forward_fm_decoder(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute velocity.
        Args:
            t:  A tensor of shape (N, 1, 1) or a tensor of a float,
                in the range of (0, 1).
            xt: the input of the current timestep, including condition
                embeddings and noisy acoustic features.
            text_condition: the text condition embeddings, with the
                shape (batch, seq_len, emb_dim).
            speech_condition: the speech condition embeddings, with the
                shape (batch, seq_len, emb_dim).
            padding_mask: The mask for padding, True means masked
                position, with the shape (N, T).
            guidance_scale: The guidance scale in classifier-free guidance,
                which is a tensor of shape (N, 1, 1) or a tensor of a float.

        Returns:
            predicted velocity, with the shape (batch, seq_len, emb_dim).
        """
        assert t.dim() in (0, 3)
        # Handle t with the shape (N, 1, 1):
        # squeeze the last dimension if it's size is 1.
        while t.dim() > 1 and t.size(-1) == 1:
            t = t.squeeze(-1)
        if guidance_scale is not None:
            while guidance_scale.dim() > 1 and guidance_scale.size(-1) == 1:
                guidance_scale = guidance_scale.squeeze(-1)
        # Handle t with a single value: expand to the size of batch size.
        if t.dim() == 0:
            t = t.repeat(xt.shape[0])
        if guidance_scale is not None and guidance_scale.dim() == 0:
            guidance_scale = guidance_scale.repeat(xt.shape[0])

        xt = torch.cat([xt, text_condition, speech_condition], dim=2)
        vt = self.fm_decoder(
            x=xt, t=t, padding_mask=padding_mask, guidance_scale=guidance_scale
        )
        return vt
    
    def forward_text_embed(
        self,
        tokens: List[List[int]],
    ):
        """
        Get the text embeddings.
        Args:
            tokens: a list of list of token ids.
        Returns:
            embed: the text embeddings, shape (batch, seq_len, emb_dim).
            tokens_lens: the length of each token sequence, shape (batch,).
        """
        device = (
            self.device if isinstance(self, DDP) else next(self.parameters()).device
        )
        tokens_padded = pad_labels(tokens, pad_id=self.pad_id, device=device)  # (B, S)
        embed = self.embed(tokens_padded)  # (B, S, C)
        tokens_lens = torch.tensor(
            [len(token) for token in tokens], dtype=torch.int64, device=device
        )
        tokens_padding_mask = make_pad_mask(tokens_lens, embed.shape[1])  # (B, S)

        embed = self.text_encoder(
            x=embed, t=None, padding_mask=tokens_padding_mask
        )  # (B, S, C)
        return embed, tokens_lens

    def forward_text_condition(
        self,
        embed: torch.Tensor,
        tokens_lens: torch.Tensor,
        features_lens: torch.Tensor,
    ):
        """
        Get the text condition with the same length of the acoustic feature.
        Args:
            embed: the text embeddings, shape (batch, token_seq_len, emb_dim).
            tokens_lens: the length of each token sequence, shape (batch,).
            features_lens: the length of each acoustic feature sequence,
                shape (batch,).
        Returns:
            text_condition: the text condition, shape
                (batch, feature_seq_len, emb_dim).
            padding_mask: the padding mask of text condition, shape
                (batch, feature_seq_len).
        """

        num_frames = int(features_lens.max())

        padding_mask = make_pad_mask(features_lens, max_len=num_frames)  # (B, T)

        tokens_durations = prepare_avg_tokens_durations(features_lens, tokens_lens)

        tokens_index = get_tokens_index(tokens_durations, num_frames).to(
            embed.device
        )  # (B, T)

        text_condition = torch.gather(
            embed,
            dim=1,
            index=tokens_index.unsqueeze(-1).expand(
                embed.size(0), num_frames, embed.size(-1)
            ),
        )  # (B, T, F)
        return text_condition, padding_mask

    def forward_text_train(
        self,
        tokens: List[List[int]],
        features_lens: torch.Tensor,
    ):
        """
        Process text for training, given text tokens and real feature lengths.
        """
        embed, tokens_lens = self.forward_text_embed(tokens)
        text_condition, padding_mask = self.forward_text_condition(
            embed, tokens_lens, features_lens
        )
        return (
            text_condition,
            padding_mask,
        )

    def forward_text_inference_gt_duration(
        self,
        tokens: List[List[int]],
        features_lens: torch.Tensor,
        prompt_tokens: List[List[int]],
        prompt_features_lens: torch.Tensor,
    ):
        """
        Process text for inference, given text tokens, real feature lengths and prompts.
        """
        tokens = [
            prompt_token + token for prompt_token, token in zip(prompt_tokens, tokens)
        ]
        features_lens = prompt_features_lens + features_lens
        embed, tokens_lens = self.forward_text_embed(tokens)
        text_condition, padding_mask = self.forward_text_condition(
            embed, tokens_lens, features_lens
        )
        return text_condition, padding_mask

    def forward_text_inference_ratio_duration(
        self,
        tokens: List[List[int]],
        prompt_tokens: List[List[int]],
        prompt_features_lens: torch.Tensor,
        speed: float,
    ):
        """
        Process text for inference, given text tokens and prompts,
        feature lengths are predicted with the ratio of token numbers.
        """
        device = (
            self.device if isinstance(self, DDP) else next(self.parameters()).device
        )

        cat_tokens = [
            prompt_token + token for prompt_token, token in zip(prompt_tokens, tokens)
        ]

        prompt_tokens_lens = torch.tensor(
            [len(token) for token in prompt_tokens],
            dtype=torch.int64,
            device=device,
        )

        tokens_lens = torch.tensor(
            [len(token) for token in tokens],
            dtype=torch.int64,
            device=device,
        )

        cat_embed, cat_tokens_lens = self.forward_text_embed(cat_tokens)

        features_lens = prompt_features_lens + torch.ceil(
            (prompt_features_lens / prompt_tokens_lens * tokens_lens / speed)
        ).to(dtype=torch.int64)

        text_condition, padding_mask = self.forward_text_condition(
            cat_embed, cat_tokens_lens, features_lens
        )
        return text_condition, padding_mask
