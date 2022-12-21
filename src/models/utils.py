from torch import nn
from functools import partial


def _apply_conv_hook(function, module: nn.Module):
    name = module.__class__.__name__
    if "Conv" in name:
        function(module)


def conv_hook(function):
    return partial(_apply_conv_hook, function)


def _init_weight(module: nn.Module, mean: float, std: float):
    nn.init.normal_(module.weight.data, mean, std)


def init_weights(mean: float, std: float):
    return partial(_init_weight, mean=mean, std=std)


def get_padding(kernel_size: int, dilation: int = 1):
    return (kernel_size * dilation - dilation) // 2
