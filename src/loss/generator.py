import torch.nn as nn
from torch import Tensor
from typing import List


class GeneratorLoss(nn.Module):
    def forward(self, fake_outs: List[Tensor]) -> Tensor:
        loss = 0
        for fake in fake_outs:
            loss = loss + ((fake - 1) ** 2).mean()  # Classify fake as 1 (real)

        return loss
