from TTS.tts.layers.glow_tts.transformer import RelativePositionTransformer
from TTS.tts.layers.glow_tts.glow import WN
from torch import nn
import torch


class TransformerResidualCouplingLayer(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 num_layers,
                 dropout_p=0,
                 cond_channels=0,
                 mean_only=False,
                 use_transformer=True
                 ):
        assert channels % 2 == 0, "Channels must be even for coupling block."
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        # input layer
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        # pre_transform layer
        self.pre_transformer = RelativePositionTransformer(
            in_channels=self.half_channels,
            out_channels=self.half_channels,
            hidden_channels=self.half_channels,
            hidden_channels_ffn=self.half_channels,
            num_heads=8,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            layer_norm_type="2",
            rel_attn_window_size=4,
        ) if use_transformer else None
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
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, m, logs, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        m0, m1 = torch.split(m, [self.half_channels] * 2, 1)
        logs0, logs1 = torch.split(logs, [self.half_channels] * 2, 1)
        x0_ = x0
        if self.pre_transformer is not None:
            x0_ = self.pre_transformer(x0 * x_mask, x_mask)
            x0_ = x0_ + x0  # residual connection
        h = self.pre(x0_) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m_flow, logs_flow = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m_flow = stats
            logs_flow = torch.zeros_like(m)

        if reverse:
            x1 = (x1 - m_flow) * torch.exp(-logs_flow) * x_mask
            m1 = (m1 - m_flow) * torch.exp(-logs_flow) * x_mask
            logs1 = logs1 - logs_flow

            x = torch.cat([x0, x1], 1)
            m = torch.cat([m0, m1], 1)
            logs = torch.cat([logs0, logs1], 1)
            return x, m, logs
        else:
            x1 = m_flow + x1 * torch.exp(logs_flow) * x_mask
            m1 = m_flow + m1 * torch.exp(logs_flow) * x_mask
            logs1 = logs1 + logs_flow

            x = torch.cat([x0, x1], 1)
            m = torch.cat([m0, m1], 1)
            logs = torch.cat([logs0, logs1], 1)
            return x, m, logs


class Flip(nn.Module):
    def forward(self, x, m, logs, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        m = torch.flip(m, [1])
        logs = torch.flip(logs, [1])
        return x, m, logs


class TransformerResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 num_layers,
                 cond_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.cond_channels = cond_channels

        self.flows = nn.ModuleList()
        for i in range(num_layers):
            use_transformer = True if (i == self.num_layers - 1) else False
            self.flows.append(TransformerResidualCouplingLayer(channels=channels,
                                                               hidden_channels=hidden_channels,
                                                               kernel_size=kernel_size,
                                                               dilation_rate=dilation_rate,
                                                               num_layers=num_layers,
                                                               cond_channels=cond_channels,
                                                               use_transformer=use_transformer))
            self.flows.append(Flip())

    def forward(self, x, m, logs, x_mask, g=None, reverse=False):
        if reverse:
            for flow in reversed(self.flows):
                x, m, logs = flow(x, m, logs, x_mask, g=g, reverse=reverse)
        else:
            for flow in self.flows:
                x, m, logs = flow(x, m, logs, x_mask, g=g, reverse=reverse)
        return x, m, logs
