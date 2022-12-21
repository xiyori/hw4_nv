import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MelDataset(Dataset):
    """Mel dataset. Read mels from numpy dump.

    Args:
        root (str): path to mels folder

    """

    def __init__(
            self,
            root
    ):
        self.root = root[:-1] if root[-1] == "/" else root

        self.ids = os.listdir(self.root)
        self.ids.sort()
        self.ids = [self.root + "/" + name for name in self.ids]

    def __getitem__(self, i):
        # Load mel
        mel = np.load(self.ids[i])

        return torch.from_numpy(mel)

    def __len__(self):
        return len(self.ids)
