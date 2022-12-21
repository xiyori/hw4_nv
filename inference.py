import argparse
import os

from torch.nn.utils import remove_weight_norm
from torch.utils.data import DataLoader
from scipy.io.wavfile import write

import sys
sys.path.append(".")

from src import config
from src.dataset import MelDataset
from src.transforms import PitchShift
from src.loops import predict
from src.utils import checkpoint


WAV_MAX = 32768


def main():
    args = parse_args()

    config_ = getattr(config, args.config)

    mel_config = config_.MelConfig()
    data_config = config_.DataConfig()
    model_config = config_.ModelConfig()
    train_config = config_.TrainConfig()

    mel_spectrogram = mel_config.mel_spectrogram.to(train_config.device)

    if args.mode == "audio":
        data_config.wav_dir = args.data_dir
        data_config.valid_file_list = os.listdir(args.data_dir)
        dataset = data_config.valid_dataset
        if args.shift != 0:
            dataset.augmentations = PitchShift(mel_config.sample_rate, args.shift)
    else:
        dataset = MelDataset(args.data_dir)

    predict_loader = DataLoader(
        dataset,
        batch_size=train_config.valid_batch,
        shuffle=False
    )

    generator = model_config.generator
    generator = generator.to(train_config.device)

    pretrained = args.gen_pretrained if args.gen_pretrained != "auto" \
                                     else train_config.run_name
    checkpoint.load_pretrained(pretrained, "generator", generator)

    generator.apply_conv(remove_weight_norm)

    predicted = predict(mel_spectrogram, generator, predict_loader,
                        train_config.device, mode = args.mode)
    save_dir = f"{args.save_dir}/{pretrained}"

    os.makedirs(save_dir, exist_ok=True)

    for name, audio in zip(data_config.valid_file_list, predicted):
        path = f"{save_dir}/{name}"
        write(path, mel_config.sample_rate, (audio.squeeze() * WAV_MAX).numpy().astype("int16"))


def parse_args():
    parser = argparse.ArgumentParser(description="Process mels or audio using trained HiFi-GAN model.")
    parser.add_argument("-c", "--config", metavar="CONFIG", type=str, default="v1",
                        help="Config filename (default: %(default)s).")
    parser.add_argument("-d", "--data_dir", type=str, default="resources/test_audio",
                        help="Path to directory with input mels or audio (default: %(default)s).")
    parser.add_argument("-s", "--save_dir", type=str, default="resources/predicted",
                        help="Where to save images (default: %(default)s).")
    parser.add_argument("-pg", "--gen_pretrained", type=str, default="auto",
                        help="Pretrained generator weights (default: %(default)s).")
    parser.add_argument("-m", "--mode", type=str, choices=["audio", "mels"], default="audio",
                        help="Type of input data (default: %(default)s).")
    parser.add_argument("--shift", type=float, default=0,
                        help="Audio pitch shift factor (default: %(default)s).")
    return parser.parse_args()


if __name__ == "__main__":
    main()
