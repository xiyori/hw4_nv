import torch
import numpy as np
import wandb

from torch import nn
from typing import Dict
from collections import defaultdict
from tqdm.auto import tqdm

from ..loss import compute_losses, compute_total_loss
from ..utils import checkpoint, pad_to_length, image_grid


def get_n_best_metric(metrics, n_best):
    return np.partition(metrics[:-1], -n_best)[-n_best]


def get_log(all_losses, gen_loss, dis_loss, metric, pred_mel, target_mel, prefix) -> Dict:
    log = {f"{prefix}_{key}": value.item() for key, (_, value) in all_losses.items()}
    log[f"{prefix}_total_gen_loss"] = gen_loss.item()
    log[f"{prefix}_total_dis_loss"] = dis_loss.item()
    with torch.no_grad():
        for key, m in metric.items():
            log[f"{prefix}_{key}"] = m(pred_mel.unsqueeze(1), target_mel.unsqueeze(1)).item()
    return log


def train(config, mel_spectrogram, mel_spectrogram_loss, train_loader, valid_loader,
          generator, discriminator, gen_optimizer, dis_optimizer,
          gen_scheduler = None, dis_scheduler = None, resume = False, progress = "epochs"):
    """
    Train model.

    Args:
        config (TrainConfig): config class containing all necessary training parameters
        mel_spectrogram (MelSpectrogram): mel spectrogram transform for input calculation
        mel_spectrogram_loss (MelSpectrogram): mel spectrogram transform for loss calculation
        train_loader (torch.utils.data.Dataloader): dataloader for train set
        valid_loader (torch.utils.data.Dataloader): dataloader for valid set
        generator (nn.Module): model to be fitted
        discriminator (nn.Module): discriminator model
        gen_optimizer (torch.optim.Optimizer): model optimizer
        dis_optimizer (torch.optim.Optimizer): discriminator optimizer
        gen_scheduler (torch.optim.lr_scheduler.): generator optimizer scheduler
        dis_scheduler (torch.optim.lr_scheduler.): discriminator optimizer scheduler
        resume (bool): resume training from the last checkpoint
        progress (str): {"epochs", "samples"} how to draw progressbars

    Returns:
        str: {"success", "gradient explosion"} exit status

    """

    super_criterion = config.super_loss
    gen_criterion = config.gen_loss
    dis_criterion = config.dis_loss

    metric = config.metric
    if isinstance(metric, nn.Module):
        metric = {config.valid_metric_name: metric}

    if resume:
        start_epoch, valid_metric_history = checkpoint.load(
            config.run_name, generator, gen_optimizer, gen_scheduler,
            discriminator, dis_optimizer, dis_scheduler
        )
    else:
        start_epoch = 0
        valid_metric_history = []

    dis_loss = 1  # Init value for use in stepper
    skip_dis_step = False
    steps_skipped = 0
    steps_waited = 0

    epoch_iter = range(start_epoch + 1, config.num_epochs + 1)
    if progress == "epochs":
        epoch_iter = tqdm(epoch_iter, desc="Epoch")

    for epoch in epoch_iter:
        # Train
        generator.train()
        discriminator.train()

        train_iter = train_loader
        if progress == "samples":
            train_iter = tqdm(train_iter, desc=f"Train {epoch}/{config.num_epochs}")

        for target_audio in train_iter:
            target_audio = target_audio.to(config.device)
            input_mel = mel_spectrogram(target_audio.squeeze(1))
            target_mel = mel_spectrogram_loss(target_audio.squeeze(1))

            # Get generator predictions
            pred_audio = generator(input_mel)
            pred_mel = mel_spectrogram_loss(pred_audio.squeeze(1))

            # Pad targets to match predictions from generator
            target_audio = pad_to_length(target_audio, pred_audio.shape[-1])
            target_mel = pad_to_length(target_mel, pred_mel.shape[-1])

            # Discriminator step
            dis_optimizer.zero_grad()

            # Skip discriminator update if the loss is too low
            if dis_loss < config.min_dis_loss:
                if not skip_dis_step:
                    steps_waited += 1
                if steps_waited >= config.steps_to_wait:
                    skip_dis_step = True
                    steps_skipped = 0
            elif skip_dis_step:
                steps_skipped += 1
                if steps_skipped >= config.steps_to_skip:
                    skip_dis_step = False
                    steps_waited = 0

            # Discriminator update
            if not skip_dis_step:
                discriminator.requires_grad(True)

                dis_out = discriminator(target_audio, pred_audio.detach())
                dis_losses = compute_losses(dis_criterion, dis_out)
                dis_loss = compute_total_loss(dis_losses)
                dis_loss.backward()

                if config.dis_grad_clip_threshold is not None:
                    nn.utils.clip_grad_norm_(discriminator.parameters(),
                                             config.dis_grad_clip_threshold)
                dis_optimizer.step()

            # Generator step
            gen_optimizer.zero_grad()

            # Turn off discriminator gradients
            discriminator.requires_grad(False)

            # Get discriminator predictions
            dis_out = discriminator(target_audio, pred_audio)

            # Generator update
            super_losses = compute_losses(super_criterion, target_mel, pred_mel)
            gen_losses = compute_losses(gen_criterion, dis_out)
            gen_loss = compute_total_loss({**super_losses, **gen_losses})

            if (torch.isnan(gen_loss) or torch.isinf(gen_loss) or
                    torch.isnan(dis_loss) or torch.isinf(dis_loss)):
                return "gradient explosion"

            gen_loss.backward()

            if config.gen_grad_clip_threshold is not None:
                nn.utils.clip_grad_norm_(generator.parameters(),
                                         config.gen_grad_clip_threshold)
            gen_optimizer.step()

            # Logger
            with torch.no_grad():
                dis_losses = compute_losses(dis_criterion, dis_out)
                dis_loss = compute_total_loss(dis_losses)
            all_losses = {**super_losses, **gen_losses, **dis_losses}

            log = get_log(all_losses, gen_loss, dis_loss, metric, pred_mel, target_mel, "train")
            log["discriminator_enabled"] = int(not skip_dis_step)

            wandb.log(log)

        # Valid
        valid_log = defaultdict(float)
        generator.eval()
        discriminator.eval()

        valid_iter = valid_loader
        if progress == "samples":
            valid_iter = tqdm(valid_iter, desc=f"Valid {epoch}/{config.num_epochs}")

        with torch.no_grad():
            for target_audio in valid_iter:
                target_audio = target_audio.to(config.device)
                input_mel = mel_spectrogram(target_audio.squeeze(1))
                target_mel = mel_spectrogram_loss(target_audio.squeeze(1))

                # Get generator predictions
                pred_audio = generator(input_mel)
                pred_mel = mel_spectrogram_loss(pred_audio.squeeze(1))

                # Pad targets to match predictions from generator
                target_audio = pad_to_length(target_audio, pred_audio.shape[-1])
                target_mel = pad_to_length(target_mel, pred_mel.shape[-1])

                # Get discriminator predictions
                dis_out = discriminator(target_audio, pred_audio.detach())
                dis_losses = compute_losses(dis_criterion, dis_out)
                dis_loss = compute_total_loss(dis_losses)

                # Compute losses
                super_losses = compute_losses(super_criterion, target_mel, pred_mel)
                gen_losses = compute_losses(gen_criterion, dis_out)
                gen_loss = compute_total_loss({**super_losses, **gen_losses})

                # Logger
                all_losses = {**super_losses, **gen_losses, **dis_losses}
                log = get_log(all_losses, gen_loss, dis_loss, metric, pred_mel, target_mel, "valid")
                for key, value in log.items():
                    valid_log[key] += value

        for key in valid_log:
            valid_log[key] /= len(valid_loader)

        valid_log = dict(valid_log)
        valid_log["mel_sample"] = wandb.Image(
            image_grid(target_mel, pred_mel, num_images=1).cpu().numpy()[::-1]
        )
        valid_log["audio_sample"] = wandb.Audio(pred_audio.squeeze(1).cpu().numpy().T,
                                                sample_rate=mel_spectrogram.sample_rate)
        valid_log["gen_lr"] = gen_optimizer.param_groups[0]["lr"]
        valid_log["dis_lr"] = dis_optimizer.param_groups[0]["lr"]
        valid_log["epoch"] = epoch

        wandb.log(valid_log, commit=False)

        # Schedulers step
        valid_metric_history += [valid_log["valid_" + config.valid_metric_name]]

        if config.provide_metric_to_scheduler:
            if gen_scheduler is not None:
                gen_scheduler.step(valid_metric_history[-1])
            if dis_scheduler is not None:
                dis_scheduler.step(valid_metric_history[-1])
        else:
            if gen_scheduler is not None:
                gen_scheduler.step()
            if dis_scheduler is not None:
                dis_scheduler.step()

        # Save model if higher metric is achieved
        if (epoch <= config.n_best_save or valid_metric_history[-1] >
                get_n_best_metric(valid_metric_history, config.n_best_save)):
            checkpoint.save_model(config.run_name, epoch, generator)

        # Save full checkpoint
        checkpoint.save(config.run_name, epoch, valid_metric_history,
                        generator, gen_optimizer, gen_scheduler,
                        discriminator, dis_optimizer, dis_scheduler)

    print("Best valid %s:  %.3f on epoch %d" %
          (config.valid_metric_name, max(valid_metric_history), np.argmax(valid_metric_history) + 1))

    return "success"
