import torch.nn as nn
from torch import Tensor
from typing import List


class DiscriminatorLoss(nn.Module):
    def forward(self, real_outs: List[Tensor], fake_outs: List[Tensor]) -> Tensor:
        loss = 0
        for real, fake in zip(real_outs, fake_outs):
            # Classify real as 1 (real), fake as 0 (fake)
            loss = loss + ((1 - real) ** 2).mean() + (fake ** 2).mean()

        return loss
