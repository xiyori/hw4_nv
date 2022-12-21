import librosa
import torch
import torchaudio
import torch.nn.functional as F

from torch import nn, Tensor


class MelSpectrogram(nn.Module):
    def __init__(self, config, loss = False):
        super().__init__()
        self.config = config
        self.sample_rate = config.sample_rate
        self.padding = (config.num_fft - config.hop_length) // 2

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.num_fft,
            f_min=config.f_min,
            f_max=config.f_max_loss if loss else config.f_max,
            n_mels=config.num_mels,
            center=False
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sample_rate,
            n_fft=config.num_fft,
            n_mels=config.num_mels,
            fmin=config.f_min,
            fmax=config.f_max_loss if loss else config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: Tensor) -> Tensor:
        """
        :param audio: Expected shape is [B, 1, T]
        :return: Shape is [B, n_mels, T']
        """

        # This is effectively the same fix, just a bit prettier
        audio = F.pad(audio, (self.padding, self.padding), mode='reflect')
        audio = audio.squeeze(1)

        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel
