import torch
import scipy.signal
import numpy as np
import math

from config import InputConfig, wavesim_duration
from wavesim_params import wavesim_field_size, wavesim_speed_of_sound, wavesim_emitter_locations, wavesim_receiver_locations

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

def crop_audio_from_location(received_signals, input_config, sample_y, sample_x):
    assert isinstance(received_signals, torch.Tensor)
    assert isinstance(input_config, InputConfig)

    receiver_indices = input_config.receiver_config.indices

    assert received_signals.shape == (len(receiver_indices), wavesim_duration)
    assert input_config.tof_cropping

    window_size = input_config.tof_crop_size
    assert window_size is not None

    sample_y *= wavesim_field_size
    sample_x *= wavesim_field_size

    c = wavesim_speed_of_sound

    # HACK: only using middle emitter for now
    assert input_config.emitter_config.indices == [2]
    emitter_y, emitter_x = wavesim_emitter_locations[2]
    distance_emitter_to_location = math.hypot(emitter_y - sample_y, emitter_x - sample_x)

    windowed_signals = []

    for signal_index, receiver_index in enumerate(receiver_indices):
        receiver_y, receiver_x = wavesim_receiver_locations[receiver_index]
        distance_location_to_receiver = math.hypot(receiver_y - sample_y, receiver_x - sample_x)

        total_distance = distance_emitter_to_location + distance_location_to_receiver

        expected_time_of_flight = total_distance / c

        window_center = int(expected_time_of_flight)
        window_start = window_center - window_size // 2
        window_end = window_center + window_size // 2

        start_padding_size = max(-window_start, 0)
        end_padding_size = max(window_end - wavesim_duration, 0)
        
        window_start = max(window_start, 0)
        window_end = min(window_end, wavesim_duration)

        signal = received_signals[signal_index]
        window = torch.cat((
            torch.zeros(start_padding_size, device=signal.device),
            signal[window_start:window_end],
            torch.zeros(end_padding_size, device=signal.device)
        ), dim=0)

        windowed_signals.append(window)
    
    windowed_signals = torch.stack(windowed_signals, dim=0)

    assert windowed_signals.shape == (len(receiver_indices), window_size)

    return windowed_signals

def crop_audio_from_location_batch(received_signals_batch, input_config, locations_yx_batch):
    assert isinstance(received_signals_batch, torch.Tensor)
    assert isinstance(input_config, InputConfig)
    assert isinstance(locations_yx_batch, torch.Tensor)
    B = received_signals_batch.shape[0]
    assert received_signals_batch.shape == (B, len(input_config.receiver_config.indices), wavesim_duration)
    assert locations_yx_batch.shape == (B, 2)
    assert input_config.tof_cropping
    assert input_config.tof_crop_size is not None
    cropped = []
    for b in range(B):
        cropped.append(crop_audio_from_location(
            received_signals_batch[b],
            input_config,
            locations_yx_batch[b, 0].item(),
            locations_yx_batch[b, 1].item()
        ))
    cropped = torch.stack(cropped, dim=0)
    assert cropped.shape == (B, len(input_config.receiver_config.indices), input_config.tof_crop_size)
    return cropped
