import torch
import torch.nn.functional as F

from torch import nn, Tensor

from .utils import get_padding, conv_hook


class ResBlockType1(nn.Module):
    def __init__(self, num_channels, kernel_size = 3, dilation = (1, 3, 5), lrelu_slope = 0.1):
        super().__init__()
        self.lrelu_slope = lrelu_slope

        self.convs1 = nn.ModuleList([
            nn.Conv1d(num_channels, num_channels, kernel_size, dilation=d,
                      padding=get_padding(kernel_size, d))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(num_channels, num_channels, kernel_size, dilation=1,
                      padding=get_padding(kernel_size, 1))
            for _ in dilation
        ])

    def forward(self, x: Tensor) -> Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = conv1(xt)
            xt = F.leaky_relu(xt, self.lrelu_slope)
            xt = conv2(xt)
            x = xt + x
        return x


class ResBlockType2(nn.Module):
    def __init__(self, num_channels, kernel_size=3, dilation=(1, 3), lrelu_slope = 0.1):
        super().__init__()
        self.lrelu_slope = lrelu_slope

        self.convs = nn.ModuleList([
            nn.Conv1d(num_channels, num_channels, kernel_size, dilation=d,
                      padding=get_padding(kernel_size, d))
            for d in dilation
        ])

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = conv(xt)
            x = xt + x
        return x


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_blocks = len(config.upsample_rates)
        self.lrelu_slope = config.lrelu_slope

        self.in_conv = nn.Conv1d(config.in_channels, config.upsample_initial_channel, kernel_size=7, padding=7 // 2)
        resblock = ResBlockType1 if config.resblock_type == 1 else ResBlockType2

        upsample_blocks = []
        channels = [config.upsample_initial_channel // 2 ** i
                    for i in range(self.num_blocks + 1)]
        for i, (stride, kernel_size) in enumerate(zip(config.upsample_rates,
                                                      config.upsample_kernel_sizes)):
            upsample_blocks += [
                nn.ConvTranspose1d(channels[i], channels[i + 1],
                                   kernel_size, stride, padding=(kernel_size - stride) // 2)
            ]
        self.upsample_blocks = nn.ModuleList(upsample_blocks)

        res_blocks = []
        for i in range(self.num_blocks):
            mrf = []
            for kernel_size, dilation in zip(config.resblock_kernel_sizes,
                                             config.resblock_dilation_sizes):
                mrf += [resblock(channels[i + 1], kernel_size, dilation, config.lrelu_slope)]
            res_blocks += [nn.ModuleList(mrf)]
        self.res_blocks = nn.ModuleList(res_blocks)

        self.out_conv = nn.Conv1d(channels[-1], 1, kernel_size=7, padding=7 // 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)
        for upsample_block, mrf in zip(self.upsample_blocks,
                                       self.res_blocks):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = upsample_block(x)
            mrf_out = None
            for res_block in mrf:
                if mrf_out is None:
                    mrf_out = res_block(x)
                else:
                    mrf_out += res_block(x)
            x = mrf_out / self.num_kernels
        x = F.leaky_relu(x)
        x = self.out_conv(x)
        x = torch.tanh(x)

        return x

    def apply_conv(self, function):
        self.apply(conv_hook(function))
