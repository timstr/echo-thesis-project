import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from assert_eq import assert_eq
from dataset3d import k_obstacles, k_sensor_recordings
from device_dict import DeviceDict
from current_simulation_description import Nx, Ny, Nz, minimum_x_units


def subset_recordings_dict(dd, sensor_indices):
    assert isinstance(dd, DeviceDict)
    assert isinstance(sensor_indices, list)
    dd_new = DeviceDict({})
    for k, v in dd.items():
        dd_new[k] = v
    sensor_recordings = dd[k_sensor_recordings]
    assert sensor_recordings.ndim in [2, 3]
    if sensor_recordings.ndim == 2:
        sensor_recordings = sensor_recordings[sensor_indices]
    elif sensor_recordings.ndim == 3:
        sensor_recordings = sensor_recordings[:, sensor_indices]
    dd_new[k_sensor_recordings] = sensor_recordings
    return dd_new


def convolve_recordings_dict(dd, emitter_signal):
    assert isinstance(dd, DeviceDict)
    assert isinstance(sensor_indices, list)
    dd_new = DeviceDict({})
    for k, v in dd.items():
        dd_new[k] = v
    sensor_recordings = dd[k_sensor_recordings]
    assert sensor_recordings.ndim in [2, 3]
    sensor_recordings = convolve_recordings(emitter_signal, sensor_recordings)
    dd_new[k_sensor_recordings] = sensor_recordings
    return dd_new


# batvision waveform input
# dd{audio} => {audio}, assert 4 channels, resample from 2048 to 3200 samples
def wavesim_to_batvision_waveform(dd):
    assert isinstance(dd, DeviceDict)
    audio = dd[k_sensor_recordings]
    batch_mode = audio.ndim == 3
    if not batch_mode:
        audio = audio.unsqueeze(0)
    B, C, L = audio.shape
    assert_eq(C, 4)
    assert_eq(L, 2048)
    audio_resampled = F.interpolate(
        audio, size=3200, mode="linear", align_corners=False
    )
    assert_eq(audio_resampled.shape, (B, 4, 3200))
    if not batch_mode:
        audio_resampled = audio_resampled.squeeze(0)
    return audio_resampled


# batvision spectrogram input
# dd{audio} => {spectrograms}, assert 4 channels, compute 4x RGB spectrogram
to_spectrogram_batvision = torchaudio.transforms.Spectrogram(
    n_fft=430,
    win_length=64,
    hop_length=6,
    window_fn=torch.hann_window,
)


def wavesim_to_batvision_spectrogram(dd):
    assert isinstance(dd, DeviceDict)
    audio = dd[k_sensor_recordings]
    batch_mode = audio.ndim == 3
    if not batch_mode:
        audio = audio.unsqueeze(0)
    B, C, L = audio.shape
    assert_eq(C, 4)
    assert_eq(L, 2048)
    spectrograms = to_spectrogram_batvision(audio)
    assert_eq(spectrograms.shape, (B, 4, 216, 342))
    spectrograms = spectrograms[:, :, :, :334]
    spectrograms = torch.log(torch.clamp(torch.abs(spectrograms), min=1e-12))
    vmin = torch.min(torch.min(spectrograms, dim=-1)[0], dim=-1)[0]
    vmax = torch.max(torch.max(spectrograms, dim=-1)[0], dim=-1)[0]
    assert_eq(vmin.shape, (B, 4))
    assert_eq(vmax.shape, (B, 4))
    vmin = vmin.reshape(B, 4, 1, 1)
    vmax = vmax.reshape(B, 4, 1, 1)
    spectrograms = (spectrograms - vmin) / (vmax - vmin)
    assert_eq(spectrograms.shape, (B, 4, 216, 334))
    if not batch_mode:
        spectrograms = spectrograms.squeeze(0)
    return spectrograms


# batvision depthmap output
# dd{obstacles} => depthmap, volume render 69x69 image with depth normalized to [0,1], resample to 128x128
def wavesim_to_batvision_depthmap(dd):
    assert isinstance(dd, DeviceDict)
    obstacles = dd[k_obstacles]
    batch_mode = obstacles.ndim == 4
    if not batch_mode:
        obstacles = obstacles.unsqueeze(0)
    B = obstacles.shape[0]
    assert_eq(obstacles.shape, (B, Nx, Ny, Nz))

    depthmap = torch.ones((B, Ny, Nz), device=obstacles.device)

    for x in range(Nx - minimum_x_units):
        xx = Nx - x - 1
        depth = 1.0 - x / (Nx - minimum_x_units - 1)
        depthmap[obstacles[:, xx]] = depth

    depthmap = F.interpolate(
        depthmap.unsqueeze(1),
        size=(128, 128),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    assert_eq(depthmap.shape, (B, 128, 128))

    if not batch_mode:
        depthmap = depthmap.squeeze(0)
    return depthmap


# batgnet spectrogram input
# dd{audio} => dd{spectrograms}, assert 4 channels, compute 4x long window spectrograms and 4x short window spectrograms, resample to 256x256
to_spectrogram_batgnet_sw = torchaudio.transforms.Spectrogram(
    n_fft=512,
    win_length=64,
    hop_length=8,
    window_fn=torch.hann_window,
)
to_spectrogram_batgnet_lw = torchaudio.transforms.Spectrogram(
    n_fft=512,
    win_length=256,
    hop_length=8,
    window_fn=torch.hann_window,
)


def wavesim_to_batgnet_spectrogram(dd):
    assert isinstance(dd, DeviceDict)
    audio = dd[k_sensor_recordings]
    batch_mode = audio.ndim == 3
    if not batch_mode:
        audio = audio.unsqueeze(0)
    B, C, L = audio.shape
    assert_eq(C, 4)
    assert_eq(L, 2048)
    spectrogram_lw = to_spectrogram_batgnet_lw(audio)
    assert_eq(spectrogram_lw.shape, (B, 4, 257, 257))
    spectrogram_sw = to_spectrogram_batgnet_sw(audio)
    assert_eq(spectrogram_sw.shape, (B, 4, 257, 257))
    spectrogram_lw = spectrogram_lw[:, :, :256, :256]
    spectrogram_sw = spectrogram_sw[:, :, :256, :256]
    spectrogram_lw = torch.log(torch.clamp(torch.abs(spectrogram_lw), min=1e-12))
    spectrogram_sw = torch.log(torch.clamp(torch.abs(spectrogram_sw), min=1e-12))
    spectrograms = torch.cat([spectrogram_sw, spectrogram_lw], dim=1)
    vmin = torch.min(torch.min(spectrograms, dim=-1)[0], dim=-1)[0]
    vmax = torch.max(torch.max(spectrograms, dim=-1)[0], dim=-1)[0]
    assert_eq(vmin.shape, (B, 8))
    assert_eq(vmax.shape, (B, 8))
    vmin = vmin.reshape(B, 8, 1, 1)
    vmax = vmax.reshape(B, 8, 1, 1)
    spectrograms = (spectrograms - vmin) / (vmax - vmin)
    assert_eq(spectrograms.shape, (B, 8, 256, 256))
    if not batch_mode:
        spectrograms = spectrograms.squeeze(0)
    return spectrograms


# batgnet occupancy output
# dd{obstacles} => dd{obstacles}, resample ROI to 64x64x64, back-fill
def wavesim_to_batgnet_occupancy(dd):
    assert isinstance(dd, DeviceDict)
    obstacles = dd[k_obstacles]
    batch_mode = obstacles.ndim == 4
    if not batch_mode:
        obstacles = obstacles.unsqueeze(0)
    B = obstacles.shape[0]
    assert_eq(obstacles.shape, (B, Nx, Ny, Nz))

    roi = obstacles[:, minimum_x_units:]

    occupancy = (
        F.interpolate(
            roi.unsqueeze(1).float(),
            size=(64, 64, 64),
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)
        > 0.5
    )

    assert_eq(occupancy.shape, (B, 64, 64, 64))

    mask = torch.zeros((B, 64, 64), dtype=torch.bool, device=obstacles.device)

    for x in range(64):
        mask.logical_or_(occupancy[:, x])
        occupancy[:, x] = mask

    if not batch_mode:
        occupancy = occupancy.squeeze(0)
    return occupancy
