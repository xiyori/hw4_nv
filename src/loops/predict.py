import torch

from tqdm.auto import tqdm


def predict(mel_spectrogram, model, data_loader,
            device = "cuda:0", mode = "audio"):
    """
    Run model on data and collect predictions.

    Args:
        mel_spectrogram (MelSpectrogram): mel spectrogram transform
        model (torch.nn.Module): model to be tested
        data_loader (torch.utils.data.Dataloader): dataloader with predict set
        device (torch.device): device to train on
        mode (str): {"audio", "mels"} data mode

    Returns:
        predicted audios
    """

    preds = []

    model.eval()
    for audio in tqdm(data_loader, desc="Predict"):
        audio = audio.to(device)
        if mode == "audio":
            input_mel = mel_spectrogram(audio)
        else:
            input_mel = audio
        with torch.no_grad():
            pred = model(input_mel)

        preds += [pred.cpu()]

    # preds = torch.cat(preds, dim=0)
    return preds
