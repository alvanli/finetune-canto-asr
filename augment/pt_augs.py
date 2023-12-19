import numpy as np

import torch
import torchaudio
import torchaudio.transforms as T


def get_spectrogram(
    n_fft=400,
    win_len=None,
    hop_len=None,
    power=2.0,
):
    waveform, _ = get_speech_sample()
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )
    return spectrogram(waveform)


def do_time_stretch(spec):
    spec = spec.squeeze()
    stretch = T.TimeStretch()
    base = torch.zeros((201,spec.size(1)))
    base[:80,:] = spec
    augd_base = stretch(base, np.random.choice([0.8, 0.9, 1.1, 1.2]))
    augd_spec = augd_base[:80,:].real
    a = torch.zeros_like(spec)
    if augd_spec.size(1) < spec.size(1):
        a[:80,:augd_spec.size(1)] = augd_spec
    else:
        a[:80,:] = augd_spec[:,:spec.size(1)]
    return a


def do_time_masking(spec):
    masking = T.TimeMasking(time_mask_param=25)
    augd_spec = masking(spec)
    return augd_spec


def do_freq_masking(spec):
    masking = T.FrequencyMasking(freq_mask_param=25)
    augd_spec = masking(spec)
    return augd_spec


