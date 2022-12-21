from torch import nn, Tensor
from typing import Dict, List

from .multi_period_discriminator import MultiPeriodDiscriminator
from .multi_scale_discriminator import MultiScaleDiscriminator
from .utils import conv_hook


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mpd = MultiPeriodDiscriminator(config)
        self.msd = MultiScaleDiscriminator(config)

    def forward(self, real: Tensor, fake: Tensor) -> Dict[str, List]:
        mpd_out = self.mpd(real, fake)
        msd_out = self.msd(real, fake)

        return {**mpd_out, **msd_out}

    def requires_grad(self, requires_grad: bool = True):
        if requires_grad:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for param in self.parameters():
                param.requires_grad = False

    def apply_conv(self, function):
        self.apply(conv_hook(function))
