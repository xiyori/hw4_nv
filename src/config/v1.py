from dataclasses import dataclass

from torch import nn
from torch.utils.data import Dataset
from torch.nn import L1Loss
from torch.nn.utils import weight_norm
from torch.optim.lr_scheduler import ExponentialLR

from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from typing import Optional, Dict, Tuple

from ..dataset import LJSpeechDataset
from ..transforms import MelSpectrogram
from ..models import Generator, Discriminator, init_weights
from ..loss import KeySelector, FeatureLoss, GeneratorLoss, DiscriminatorLoss


@dataclass
class MelConfig:
    sample_rate: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    num_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    f_max_loss: Optional[int] = None
    num_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251

    @property
    def mel_spectrogram(self):
        return MelSpectrogram(self)

    @property
    def mel_spectrogram_loss(self):
        return MelSpectrogram(self, loss=True)


@dataclass
class DataConfig:
    # data_dir = "C:/data/LJSpeech-1.1"
    data_dir = "/home/jupyter/mnt/datasets/LJSpeech-1.1"
    wav_dir = data_dir + "/wavs"
    train_file_list = data_dir + "/training.txt"
    valid_file_list = data_dir + "/validation.txt"

    segment_size = 8192
    wav_scale = 0.95

    @property
    def train_dataset(self) -> Dataset:
        return LJSpeechDataset(self, MelConfig(), self.train_file_list)

    @property
    def valid_dataset(self) -> Dataset:
        return LJSpeechDataset(self, MelConfig(), self.valid_file_list,
                               validation=True)


@dataclass
class ModelConfig:
    resblock_type = 1
    lrelu_slope = 0.1

    init_mean = 0.
    init_std = 0.01

    in_channels = MelConfig.num_mels
    upsample_rates = [8, 8, 2, 2]
    upsample_kernel_sizes = [16, 16, 4, 4]
    upsample_initial_channel = 512
    resblock_kernel_sizes = [3, 7, 11]
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

    # Parameters from the official implementation
    mpd_periods = [2, 3, 5, 7, 11]
    mpd_channels = [1, 32, 128, 512, 1024, 1024]
    mpd_kernel_size = 5
    mpd_stride = 3
    mpd_use_spectral_norm = False

    # Parameters from the official implementation
    msd_channels = [1, 128, 128, 256, 512, 1024, 1024, 1024]
    msd_kernel_sizes = [15, 41, 41, 41, 41, 41, 5]
    msd_strides = [1, 2, 2, 4, 4, 1, 1]
    msd_groups = [1, 4, 16, 16, 16, 16, 1]

    @property
    def generator(self) -> nn.Module:
        model = Generator(self)
        model.apply_conv(weight_norm)
        model.apply_conv(init_weights(self.init_mean, self.init_std))
        return model

    @property
    def discriminator(self) -> nn.Module:
        model = Discriminator(self)
        return model


@dataclass
class TrainConfig:
    wandb_project = "hifi-gan"

    device = "cuda:0"# if torch.cuda.is_available() else "cpu"

    num_epochs = 3100

    train_batch = 16
    valid_batch = 1

    train_num_workers = 4
    valid_num_workers = 0

    l1_mel_coef = 45
    feature_coef = 1
    gen_loss_coef = 1

    dis_loss_coef = 1
    min_dis_loss = 0.3  #: Discriminator loss lower bound
    steps_to_skip = 1   #: Steps to skip discriminator update if lower bound is reached
    steps_to_wait = 1   #: Steps to wait after lower bound is reached before skipping

    valid_metric_name = "ssim"

    gen_lr = 2e-4
    dis_lr = 2e-4
    betas = (0.8, 0.99)
    lr_gamma = 0.999

    gen_grad_clip_threshold = None
    dis_grad_clip_threshold = None

    provide_metric_to_scheduler = False
    n_best_save = 1

    def gen_scheduler(self, optimizer):
        return ExponentialLR(optimizer, gamma=self.lr_gamma)

    def dis_scheduler(self, optimizer):
        return ExponentialLR(optimizer, gamma=self.lr_gamma)

    @property
    def super_loss(self) -> Dict[str, Tuple[float, nn.Module]]:
        criterion = dict()
        criterion["l1_mel_loss"] = self.l1_mel_coef, L1Loss()
        return criterion

    @property
    def gen_loss(self) -> Dict[str, Tuple[float, nn.Module]]:
        criterion = dict()
        for prefix in ["mpd", "msd"]:
            criterion[prefix + "_feature_loss"] = self.feature_coef, KeySelector(
                prefix + "_real_features",
                prefix + "_fake_features",
                FeatureLoss()
            )
            criterion[prefix + "_gen_loss"] = self.gen_loss_coef, KeySelector(
                prefix + "_fake_outs",
                GeneratorLoss()
            )
        return criterion

    @property
    def dis_loss(self) -> Dict[str, Tuple[float, nn.Module]]:
        criterion = dict()
        for prefix in ["mpd", "msd"]:
            criterion[prefix + "_dis_loss"] = self.dis_loss_coef, KeySelector(
                prefix + "_real_outs",
                prefix + "_fake_outs",
                DiscriminatorLoss()
            )
        return criterion

    @property
    def metric(self) -> Dict[str, nn.Module]:
        return {"psnr": PeakSignalNoiseRatio().to(self.device),
                "ssim": StructuralSimilarityIndexMeasure().to(self.device)}

    @property
    def run_name(self) -> str:
        return f"v1_l1mel{self.l1_mel_coef}_genlr{self.gen_lr}" \
               f"_dislr{self.dis_lr}_batch{self.train_batch}"
