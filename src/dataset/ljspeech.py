import numpy as np
import torchaudio
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import Dataset
from typing import Sequence


class LJSpeechDataset(Dataset):
    def __init__(self, config, mel_config, file_list,
                 validation = False, augmentations = None):
        self.wav_dir = config.wav_dir
        self.wav_scale = config.wav_scale
        self.segment_size = config.segment_size

        self.sample_rate = mel_config.sample_rate
        self.hop_length = mel_config.hop_length
        self.padding = (mel_config.num_fft - mel_config.hop_length) // 2

        self.file_list = file_list
        self.validation = validation
        self.augmentations = augmentations

        if isinstance(file_list, str):
            self.ids = []
            with open(file_list, "r") as f:
                for line in f:
                    name = line[:line.index("|")].strip()
                    self.ids += [f"{self.wav_dir}/{name}.wav"]
        elif isinstance(file_list, Sequence):
            self.ids = [f"{self.wav_dir}/{name}" for name in file_list]
        else:
            raise ValueError(f"file list of type '{type(file_list)}' not supported")

    def __getitem__(self, i: int) -> Tensor:
        audio, sample_rate = torchaudio.load(self.ids[i])
        assert sample_rate == self.sample_rate
        audio = audio / audio.max(dim=1, keepdim=True)[0] * self.wav_scale

        if self.segment_size is not None and not self.validation:
            if audio.shape[1] >= self.segment_size:
                audio_start_bound = audio.shape[1] - self.segment_size
                audio_start = np.random.randint(audio_start_bound)
                audio = audio[:, audio_start:audio_start + self.segment_size]
            else:
                # I'm almost 100% sure this is never used
                audio = F.pad(audio, (0, self.segment_size - audio.shape[1]), "constant")

        if self.augmentations is not None:
            audio = self.augmentations(audio)

        return audio

    def __len__(self):
        return len(self.ids)
