import torch.nn.functional as F

from torch import nn, Tensor
from torch.nn.utils import weight_norm, spectral_norm
from typing import Dict, List, Tuple

from .utils import conv_hook


class ScaleDiscriminator(nn.Module):
    def __init__(self, channels, kernel_sizes, strides, groups,
                 use_spectral_norm = False, lrelu_slope = 0.1):
        super().__init__()
        self.lrelu_slope = lrelu_slope

        convs = []
        for i in range(len(channels) - 1):
            convs += [nn.Conv1d(channels[i], channels[i + 1],
                                kernel_sizes[i], strides[i],
                                groups=groups[i],
                                padding=kernel_sizes[i] // 2)]
        self.convs = nn.ModuleList(convs)
        self.out_conv = nn.Conv1d(channels[-1], 1, kernel_size=3, padding=3 // 2)

        norm = spectral_norm if use_spectral_norm else weight_norm
        self.apply(conv_hook(norm))

    def forward(self, x: Tensor) -> Tuple[Tensor, List]:
        feature_map = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            feature_map.append(x)
        x = self.out_conv(x)
        feature_map.append(x)
        x = x.reshape(x.shape[0], -1)

        return x, feature_map


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        args = [config.msd_channels, config.msd_kernel_sizes, config.msd_strides, config.msd_groups]
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(*args, use_spectral_norm=True, lrelu_slope=config.lrelu_slope),
            ScaleDiscriminator(*args, lrelu_slope=config.lrelu_slope),
            ScaleDiscriminator(*args, lrelu_slope=config.lrelu_slope),
        ])
        self.avgpools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2)
        ])

    def forward(self, real: Tensor, fake: Tensor) -> Dict[str, List]:
        real_outs = []
        fake_outs = []
        real_features = []
        fake_features = []
        for i, discr in enumerate(self.discriminators):
            if i > 0:
                real = self.avgpools[i - 1](real)
                fake = self.avgpools[i - 1](fake)
            real_out, real_feature_map = discr(real)
            fake_out, fake_feature_map = discr(fake)
            real_outs.append(real_out)
            real_features.append(real_feature_map)
            fake_outs.append(fake_out)
            fake_features.append(fake_feature_map)

        return {"msd_real_outs": real_outs,
                "msd_fake_outs": fake_outs,
                "msd_real_features": real_features,
                "msd_fake_features": fake_features}
