import torch.nn.functional as F
from torch import nn, Tensor


def pad_to_length(x: Tensor, length):
    padding_right = padding_left = (length - x.shape[-1]) // 2
    if (length - x.shape[-1]) % 2 == 1:
        padding_right += 1
    if padding_right > 0:
        x = F.pad(x, (padding_left, padding_right), mode='reflect')
    return x
