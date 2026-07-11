"""
QuantGAN-style GAN for synthetic financial return generation.

Architecture: Temporal Convolutional Networks (dilated causal convolutions)
for both Generator and Discriminator. TCNs handle long-range dependence
(volatility clustering, autocorrelation) much better than plain MLPs/RNNs
for financial time series.

Expected input: standardized log-return windows, shape (batch, seq_len, 1)
-- this matches what SynFinBench's preprocessing notebook already outputs.

UPDATE: Discriminator now includes a minibatch-stddev feature to combat
mode collapse (see minibatch_stddev() below).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block: one dilated causal convolution layer with residual connection
# ---------------------------------------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # "causal" padding: pad only on the left so the model never sees the future
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                padding=padding, dilation=dilation)
        self.chomp = padding  # amount to trim off the right side after conv
        self.relu = nn.LeakyReLU(0.2)

        # if channel counts differ, project the residual so shapes match
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                            if in_channels != out_channels else None)

    def _trim(self, x):
        # remove the extra padding from the right so output length == input length
        return x[:, :, :-self.chomp] if self.chomp != 0 else x

    def forward(self, x):
        out = self.relu(self._trim(self.conv1(x)))
        out = self.relu(self._trim(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilation (1,2,4,8,...)."""

    def __init__(self, in_channels, hidden_channels, n_layers, kernel_size=3):
        super().__init__()
        layers = []
        ch_in = in_channels
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(TemporalBlock(ch_in, hidden_channels, kernel_size, dilation))
            ch_in = hidden_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        return self.net(x)


# ---------------------------------------------------------------------------
# Generator: noise sequence -> fake return sequence
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim=8, hidden_channels=64, n_layers=6):
        super().__init__()
        self.tcn = TCN(in_channels=noise_dim, hidden_channels=hidden_channels, n_layers=n_layers)
        self.out = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def forward(self, z):
        # z: (batch, seq_len, noise_dim) -> conv1d wants (batch, channels, seq_len)
        z = z.transpose(1, 2)
        h = self.tcn(z)
        out = self.out(h)          # (batch, 1, seq_len)
        return out.transpose(1, 2)  # back to (batch, seq_len, 1)


# ---------------------------------------------------------------------------
# Discriminator: real or fake sequence -> single score (real-ness)
# ---------------------------------------------------------------------------
def minibatch_stddev(x):
    """
    Computes the stddev across the batch dimension and appends it as an extra
    channel. This lets the discriminator directly "see" how diverse (or
    collapsed) a batch of samples is -- if the generator starts producing
    near-identical outputs, this feature makes that trivially detectable,
    which pushes the generator away from mode collapse.
    x: (batch, channels, seq_len)
    """
    std = x.std(dim=0, keepdim=True)                        # (1, channels, seq_len)
    std_mean = std.mean().expand(x.size(0), 1, x.size(2))   # (batch, 1, seq_len)
    return torch.cat([x, std_mean], dim=1)                   # (batch, channels+1, seq_len)


class Discriminator(nn.Module):
    def __init__(self, hidden_channels=64, n_layers=6):
        super().__init__()
        # in_channels=2: the raw sequence (1) + minibatch stddev feature (1)
        self.tcn = TCN(in_channels=2, hidden_channels=hidden_channels, n_layers=n_layers)
        self.out = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        x = x.transpose(1, 2)              # (batch, 1, seq_len)
        x = minibatch_stddev(x)            # (batch, 2, seq_len)
        h = self.tcn(x)
        score = self.out(h)                # (batch, 1, seq_len)
        return score.mean(dim=[1, 2])      # average pooled score, (batch,)