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
    return np.log(np.clip(np.abs(Sxx), a_min=1e-6, a_max=None))

def make_spectrogram(audio_raw):
    dims = audio_raw.shape
    if len(dims) == 1:
        return make_spectrogram_single(audio_raw)
    elif len(dims) > 1:
        out = []
        for i in range(dims[0]):
            out.append(make_spectrogram(audio_raw[i]))
        return np.array(out)

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