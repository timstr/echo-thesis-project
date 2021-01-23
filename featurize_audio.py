import torch
import scipy.signal
import scipy.io.wavfile as wf
import numpy as np
import math

def make_spectrogram_single(audio_raw):
    assert len(audio_raw.shape) == 1
    assert audio_raw.shape[0] > 256
    window_size = 64
    num_frequency_bins = 10
    f, t, Sxx = scipy.signal.spectrogram(
        audio_raw,
        window="blackmanharris",
        nperseg=window_size,
        noverlap=window_size*7//8
    )
    # Sxx = Sxx[:num_frequency_bins,:]
    sg = np.log(np.clip(np.abs(Sxx), a_min=1e-6, a_max=128))
    measured_lower_bound = -14.0
    measured_upper_bound = 5.0
    sg_normalized = (sg - measured_lower_bound) / (measured_upper_bound - measured_lower_bound)
    return sg_normalized

def make_spectrogram(audio_raw):
    dims = audio_raw.shape
    if len(dims) == 1:
        return make_spectrogram_single(audio_raw)
    elif len(dims) > 1:
        out = []
        for i in range(dims[0]):
            out.append(make_spectrogram(audio_raw[i]))
        return np.array(out)

def make_gcc_phat_single(original, echo):
    n = 2048 # Number of audio samples in dataset version 9
    assert original.shape == (n,)
    assert echo.shape == (n,)
    # Adapted from https://github.com/xiongyihui/tdoa/blob/a52505672f15b50f1c07606e6609bb1cb016add8/gcc_phat.py under the Apache License 2.0
    sig = np.fft.rfft(echo, n=n)
    refsig = np.fft.rfft(original, n=n)
    r = sig * np.conj(refsig)
    cc = np.fft.irfft(r / np.abs(r), n=n)
    assert cc.shape == (n,)
    return cc.astype("float32")

def make_gcc_phat(original, echoes):
    n = 2048 # Number of audio samples in dataset version 9
    assert original.shape == (n,)
    assert len(echoes.shape) == 2
    assert echoes.shape[1] == n
    res = np.stack([
        make_gcc_phat_single(original, echo) for echo in echoes
    ], axis=0)
    assert res.shape == (echoes.shape[0], n)
    return res

# signed, clipped logarithm
def sclog(t):
    max_val = 1e0
    min_val = 1e-4
    signs = torch.sign(t)
    t = torch.abs(t)
    t = torch.clamp(t, min=min_val, max=max_val)
    t = torch.log(t)
    t = (t - math.log(min_val)) / (math.log(max_val) - math.log(min_val))
    t = t * signs
    return t

def normalize_amplitude(waveform):
    max_amp = torch.max(torch.abs(waveform))
    if max_amp > 1e-3:
        waveform = (0.5 / max_amp) * waveform
    return waveform