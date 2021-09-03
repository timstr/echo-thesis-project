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
to_spectrogram = torchaudio.transforms.Spectrogram(
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
    spectrogram = to_spectrogram(audio)
    assert_eq(spectrogram.shape, (B, 4, 216, 342))
    spectrogram = spectrogram[:, :, :, :334]
    spectrogram = torch.log(torch.clamp(torch.abs(spectrogram), min=1e-12))
    vmin = torch.min(spectrogram)
    vmax = torch.max(spectrogram)
    spectrogram = (spectrogram - vmin) / (vmax - vmin)
    assert_eq(spectrogram.shape, (B, 4, 216, 334))
    if not batch_mode:
        spectrogram = spectrogram.squeeze(0)
    return spectrogram


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

    roi = obstacles[:, minimum_x_units:]

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
def wavesim_to_batgnet_spectrogram(dd):
    assert isinstance(dd, DeviceDict)
    # TODO
    pass


# batgnet occupancy output
# dd{obstacles} => dd{obstacles}, resample ROI to 64x64x64, back-fill
def wavesim_to_batgnet_occupancy(dd):
    assert isinstance(dd, DeviceDict)
    # TODO
    pass


# TODO: remove after testing
if __name__ == "__main__":
    import math
    import matplotlib.pyplot as plt
    from current_simulation_description import make_receiver_indices
    from signals_and_geometry import convolve_recordings, make_fm_chirp
    from dataset3d import WaveDataset3d
    from current_simulation_description import make_simulation_description

    desc = make_simulation_description()
    fm_chirp = make_fm_chirp(
        begin_frequency_Hz=18_000.0,
        end_frequency_Hz=22_000.0,
        sampling_frequency=desc.output_sampling_frequency,
        chirp_length_samples=math.ceil(0.001 * desc.output_sampling_frequency),
        wave="sine",
    )
    sensor_indices = make_receiver_indices(num_x=1, num_y=2, num_z=2)
    with WaveDataset3d(desc, "dataset_7.5mm_random_smol.h5") as ds:
        dd = ds[9]

        dd = subset_recordings_dict(dd, sensor_indices)
        dd = convolve_recordings_dict(dd, fm_chirp)

        audio_resampled = wavesim_to_batvision_waveform(dd)
        print(f"batvision waveform:\n    {audio_resampled.shape}")

        for i in range(4):
            plt.plot(audio_resampled[i])
        plt.show()

        spectrogram = wavesim_to_batvision_spectrogram(dd)
        print(f"batvision spectrogram:\n    {spectrogram.shape}")
        plt.imshow(spectrogram[:3].permute(1, 2, 0))
        plt.show()

        depthmap = wavesim_to_batvision_depthmap(dd)
        print(f"batvision depthmap:\n    {depthmap.shape}")
        plt.imshow(depthmap, cmap="gray")
        plt.show()
