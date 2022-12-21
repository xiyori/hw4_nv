from torch import nn, Tensor
from typing import Dict, Tuple


def compute_losses(criterion: Dict[str, Tuple[float, nn.Module]],
                   *args) -> Dict[str, Tuple[float, Tensor]]:
    return {key: (coef, l(*args)) for key, (coef, l) in criterion.items()}


def compute_total_loss(losses: Dict[str, Tuple[float, Tensor]]) -> Tensor:
    return sum(map(lambda x: x[0] * x[1], losses.values()))
