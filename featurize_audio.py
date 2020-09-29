import scipy.signal
import scipy.io.wavfile as wf
import numpy as np

def make_spectrogram_single(audio_raw):
    assert(len(audio_raw.shape) == 1)
    assert(audio_raw.shape[0] > 256)
    window_size = 256
    num_frequency_bins = 10
    f, t, Sxx = scipy.signal.spectrogram(
        audio_raw,
        window="blackmanharris",
        nperseg=window_size,
        noverlap=window_size*3//4
    )
    Sxx = Sxx[:num_frequency_bins,:]
    return np.log(np.clip(np.abs(Sxx), a_min=1e-6, a_max=None))

def make_spectrogram(audio_raw):
    dims = audio_raw.shape
    if len(dims) == 1:
        return make_spectrogram_single(audio_raw)
    elif len(dims) == 2:
        out = []
        for i in range(dims[0]):
            out.append(make_spectrogram_single(audio_raw[i]))
        return np.array(out)
