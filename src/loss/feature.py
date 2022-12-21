import torch.nn as nn

from torch import Tensor
from typing import List


class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, real_features: List[Tensor], fake_features: List[Tensor]) -> Tensor:
        loss = 0
        for real_feature_map, fake_feature_map in zip(real_features, fake_features):
            for real_feature, fake_feature in zip(real_feature_map, fake_feature_map):
                loss = loss + self.l1_loss(fake_feature, real_feature)

        return loss
