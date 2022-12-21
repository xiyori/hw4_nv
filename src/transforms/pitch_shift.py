import torch
import librosa

from torch import nn, Tensor


class PitchShift(nn.Module):
    def __init__(self, sample_rate, shift = 0.):
        super().__init__()
        self.sample_rate = sample_rate
        self.shift = shift

    def forward(self, audio: Tensor) -> Tensor:
        audio = librosa.effects.pitch_shift(audio.numpy().squeeze(0),
                                            sr=self.sample_rate, n_steps=self.shift)
        audio = torch.from_numpy(audio).unsqueeze(0)

        return audio
