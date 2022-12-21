import torch.nn.functional as F

from torch import nn, Tensor
from torch.nn.utils import weight_norm, spectral_norm
from typing import Dict, List, Tuple

from .utils import conv_hook


class PeriodDiscriminator(nn.Module):
    def __init__(self, period, channels, kernel_size = 5, stride = 3,
                 use_spectral_norm = False, lrelu_slope = 0.1):
        super().__init__()
        self.period = period
        self.lrelu_slope = lrelu_slope

        convs = []
        for i in range(len(channels) - 2):
            convs += [nn.Conv2d(channels[i], channels[i + 1],
                                (kernel_size, 1), (stride, 1),
                                padding=(kernel_size // 2, 0))]
        convs += [nn.Conv2d(channels[-2], channels[-1],
                            (kernel_size, 1),
                            padding=(kernel_size // 2, 0))]
        self.convs = nn.ModuleList(convs)

        self.out_conv = nn.Conv2d(channels[-1], 1, kernel_size=(3, 1), padding=(3 // 2, 0))

        norm = spectral_norm if use_spectral_norm else weight_norm
        self.apply(conv_hook(norm))

    def forward(self, x: Tensor) -> Tuple[Tensor, List]:
        batch, channels, length = x.shape
        if length % self.period != 0:
            n_pad = self.period - (length % self.period)
            x = F.pad(x, (0, n_pad), mode="reflect")
            length += n_pad
        x = x.reshape(batch, channels, length // self.period, self.period)

        feature_map = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            feature_map.append(x)
        x = self.out_conv(x)
        feature_map.append(x)
        x = x.reshape(batch, -1)

        return x, feature_map


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        discriminators = [PeriodDiscriminator(period,
                                              config.mpd_channels,
                                              config.mpd_kernel_size,
                                              config.mpd_stride,
                                              config.mpd_use_spectral_norm,
                                              config.lrelu_slope)
                          for period in config.mpd_periods]
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, real: Tensor, fake: Tensor) -> Dict[str, List]:
        real_outs = []
        fake_outs = []
        real_features = []
        fake_features = []
        for discr in self.discriminators:
            real_out, real_feature_map = discr(real)
            fake_out, fake_feature_map = discr(fake)
            real_outs.append(real_out)
            real_features.append(real_feature_map)
            fake_outs.append(fake_out)
            fake_features.append(fake_feature_map)

        return {"mpd_real_outs": real_outs,
                "mpd_fake_outs": fake_outs,
                "mpd_real_features": real_features,
                "mpd_fake_features": fake_features}
