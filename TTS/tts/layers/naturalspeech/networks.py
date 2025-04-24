import math
import torch
from torch import nn
from TTS.tts.layers.glow_tts.glow import WN
from TTS.tts.layers.glow_tts.transformer import RelativePositionTransformer, RelativePositionMultiHeadAttention
from TTS.tts.utils.helpers import sequence_mask
from TTS.tts.layers.generic.normalization import LayerNorm
from TTS.tts.layers.naturalspeech.modules import ConvBlock, LinearNorm, SwishBlock


LRELU_SLOPE = 0.1


class DurationPredictor(nn.Module):
    def __init__(self,
                 in_channels,
                 filter_channels,
                 kernel_size,
                 p_dropout,
                 gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class LearnableUpsampling(nn.Module):
    def __init__(
        self,
        d_predictor=192,
        kernel_size=3,
        dropout=0.0,
        conv_output_size=8,
        dim_w=4,
        dim_c=2,
        max_seq_len=1000,
    ):
        super(LearnableUpsampling, self).__init__()
        self.max_seq_len = max_seq_len

        # Attention (W)
        self.conv_w = ConvBlock(
            d_predictor,
            conv_output_size,
            kernel_size,
            dropout=dropout,
            activation=nn.SiLU(),
        )
        self.swish_w = SwishBlock(conv_output_size + 2, dim_w, dim_w)
        self.linear_w = LinearNorm(dim_w * d_predictor, d_predictor, bias=True)
        self.softmax_w = nn.Softmax(dim=2)

        # Auxiliary Attention Context (C)
        self.conv_c = ConvBlock(
            d_predictor,
            conv_output_size,
            kernel_size,
            dropout=dropout,
            activation=nn.SiLU(),
        )
        self.swish_c = SwishBlock(conv_output_size + 2, dim_c, dim_c)

        # Upsampled Representation (O)
        self.linear_einsum = LinearNorm(dim_c * dim_w, d_predictor)  # A
        self.layer_norm = nn.LayerNorm(d_predictor)

        self.proj_o = LinearNorm(192, 192 * 2)

    def forward(self, duration, V, src_len, src_mask, max_src_len):

        batch_size = duration.shape[0]

        # Duration Interpretation
        mel_len = torch.round(duration.sum(-1)).type(torch.LongTensor).to(V.device)
        mel_len = torch.clamp(mel_len, max=self.max_seq_len)
        max_mel_len = mel_len.max().item()
        mel_mask = self.get_mask_from_lengths(mel_len, max_mel_len)

        # Prepare Attention Mask
        src_mask_ = src_mask.unsqueeze(1).expand(
            -1, mel_mask.shape[1], -1
        )  # [B, tat_len, src_len]
        mel_mask_ = mel_mask.unsqueeze(-1).expand(
            -1, -1, src_mask.shape[1]
        )  # [B, tgt_len, src_len]
        attn_mask = torch.zeros(
            (src_mask.shape[0], mel_mask.shape[1], src_mask.shape[1])
        ).to(V.device)
        attn_mask = attn_mask.masked_fill(src_mask_, 1.0)
        attn_mask = attn_mask.masked_fill(mel_mask_, 1.0)
        attn_mask = attn_mask.bool()

        # Token Boundary Grid
        e_k = torch.cumsum(duration, dim=1)
        s_k = e_k - duration
        e_k = e_k.unsqueeze(1).expand(batch_size, max_mel_len, -1)
        s_k = s_k.unsqueeze(1).expand(batch_size, max_mel_len, -1)
        t_arange = (
            torch.arange(1, max_mel_len + 1, device=V.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, -1, max_src_len)
        )

        S, E = (t_arange - s_k).masked_fill(attn_mask, 0), (e_k - t_arange).masked_fill(
            attn_mask, 0
        )

        # Attention (W)
        W = self.swish_w(S, E, self.conv_w(V))  # [B, T, K, dim_w]
        W = W.masked_fill(src_mask_.unsqueeze(-1), -np.inf)
        W = self.softmax_w(W)  # [B, T, K]
        W = W.masked_fill(mel_mask_.unsqueeze(-1), 0.0)
        W = W.permute(0, 3, 1, 2)

        # Auxiliary Attention Context (C)
        C = self.swish_c(S, E, self.conv_c(V))  # [B, T, K, dim_c]

        # Upsampled Representation (O)
        upsampled_rep = self.linear_w(
            torch.einsum("bqtk,bkh->bqth", W, V).permute(0, 2, 1, 3).flatten(2)
        ) + self.linear_einsum(
            torch.einsum("bqtk,btkp->bqtp", W, C).permute(0, 2, 1, 3).flatten(2)
        )  # [B, T, M]
        upsampled_rep = self.layer_norm(upsampled_rep)
        upsampled_rep = upsampled_rep.masked_fill(mel_mask.unsqueeze(-1), 0)
        upsampled_rep = self.proj_o(upsampled_rep)

        return upsampled_rep, mel_mask, mel_len, W

    def get_mask_from_lengths(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()

        ids = (
            torch.arange(0, max_len)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(lengths.device)
        )
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

        return mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        dropout_p: float,
        language_emb_dim: int = None,
    ):
        """Text Encoder for VITS model.

        Args:
            n_vocab (int): Number of characters for the embedding layer.
            out_channels (int): Number of channels for the output.
            hidden_channels (int): Number of channels for the hidden layers.
            hidden_channels_ffn (int): Number of channels for the convolutional layers.
            num_heads (int): Number of attention heads for the Transformer layers.
            num_layers (int): Number of Transformer layers.
            kernel_size (int): Kernel size for the FFN layers in Transformer network.
            dropout_p (float): Dropout rate for the Transformer layers.
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)

        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        if language_emb_dim:
            hidden_channels += language_emb_dim

        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            layer_norm_type="2",
            rel_attn_window_size=4,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, lang_emb=None):
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_length: :math:`[B]`
        """
        assert x.shape[0] == x_lengths.shape[0]
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]

        # concat the lang emb in embedding chars
        if lang_emb is not None:
            x = torch.cat((x, lang_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1)

        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)  # [b, 1, t]

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        dropout_p=0,
        cond_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        # input layer
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        # coupling layers
        self.enc = WN(
            hidden_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            dropout_p=dropout_p,
            c_in_channels=cond_channels,
        )
        # output layer
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Note:
            Set `reverse` to True for inference.

        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, log_scale = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            log_scale = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(log_scale) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(log_scale, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-log_scale) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class ResidualCouplingBlocks(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        num_flows=4,
        cond_channels=0,
    ):
        """Redisual Coupling blocks for VITS flow layers.

        Args:
            channels (int): Number of input and output tensor channels.
            hidden_channels (int): Number of hidden network channels.
            kernel_size (int): Kernel size of the WaveNet layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            num_flows (int, optional): Number of Residual Coupling blocks. Defaults to 4.
            cond_channels (int, optional): Number of channels of the conditioning tensor. Defaults to 0.
        """
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.num_flows = num_flows
        self.cond_channels = cond_channels

        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(
                ResidualCouplingBlock(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    num_layers,
                    cond_channels=cond_channels,
                    mean_only=True,
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Note:
            Set `reverse` to True for inference.

        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
                x = torch.flip(x, [1])
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, [1])
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        cond_channels=0,
    ):
        """Posterior Encoder of VITS model.

        ::
            x -> conv1x1() -> WaveNet() (non-causal) -> conv1x1() -> split() -> [m, s] -> sample(m, s) -> z

        Args:
            in_channels (int): Number of input tensor channels.
            out_channels (int): Number of output tensor channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size of the WaveNet convolution layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            cond_channels (int, optional): Number of conditioning tensor channels. Defaults to 0.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.cond_channels = cond_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels, hidden_channels, kernel_size, dilation_rate, num_layers, c_in_channels=cond_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_lengths: :math:`[B, 1]`
            - g: :math:`[B, C, 1]`
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        mean, log_scale = torch.split(stats, self.out_channels, dim=1)
        z = (mean + torch.randn_like(mean) * torch.exp(log_scale)) * x_mask
        return z, mean, log_scale, x_mask


class VAEMemoryBank(nn.Module):
    def __init__(self,
                 bank_size: int = 1000,
                 n_hidden_dims: int = 192,
                 n_attn_heads: int = 2,
                 init_values=None,):
        super().__init__()
        self.bank_size = bank_size
        self.n_hidden_dims = n_hidden_dims
        self.n_attn_heads = n_attn_heads

        self.encoder = RelativePositionMultiHeadAttention(channels=n_hidden_dims,
                                                          out_channels=n_hidden_dims,
                                                          num_heads=n_attn_heads)
        self.memory_bank = nn.Parameter(torch.randn(bank_size, n_hidden_dims))
        if init_values is not None:
            with torch.no_grad():
                self.memory_bank.copy_(init_values)

    def forward(self, z):
        b, _, _ = z.shape
        return self.encoder(z, self.memory_bank.unsqueeze(0).expand(b, 1, 1), attn_mask=None)