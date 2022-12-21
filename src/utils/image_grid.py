import torch
from torch import Tensor


def image_grid(*images, num_images = 1) -> Tensor:
    # images = [image.repeat(1, 3 // image.shape[1], 1, 1) for image in images]
    vertical_grid = torch.cat(images, dim=-2)
    images = torch.unbind(vertical_grid[:num_images], dim=0)
    horizontal_grid = torch.cat(images, dim=-1)
    # out = torch.clip(horizontal_grid, min=clip[0], max=clip[1])
    return horizontal_grid
