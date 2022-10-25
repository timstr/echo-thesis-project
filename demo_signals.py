import fix_dead_command_line

from dataset_adapters import (
    convolve_recordings_dict,
    subset_recordings_dict,
    wavesim_to_batvision_spectrogram,
)

import matplotlib.pyplot as plt
import torch
import math
import numpy as np
import scipy.fft as fft
from argparse import ArgumentParser

from current_simulation_description import (
    make_receiver_indices,
    make_simulation_description,
)
from dataset3d import WaveDataset3d, k_sensor_recordings
from which_device import get_compute_device
from signals_and_geometry import make_fm_chirp, convolve_recordings


def main():
    parser = ArgumentParser()
    parser.add_argument("path_to_dataset", type=str)
    parser.add_argument("index", nargs="?", type=int, default=0)
    parser.add_argument(
        "--f0",
        type=float,
        dest="f0",
        help="chirp start frequency (Hz)",
        default=18_000.0,
    )
    parser.add_argument(
        "--f1", type=float, dest="f1", help="chirp end frequency (Hz)", default=20_000.0
    )
    parser.add_argument(
        "--l", type=float, dest="l", help="chirp duration (seconds)", default=0.001
    )
    args = parser.parse_args()

    description = make_simulation_description()
    dataset = WaveDataset3d(description, args.path_to_dataset)

    receiver_indices = make_receiver_indices(1, 2, 2)

    chirp = make_fm_chirp(
        begin_frequency_Hz=args.f0,
        end_frequency_Hz=args.f1,
        sampling_frequency=description.output_sampling_frequency,
        chirp_length_samples=math.ceil(args.l * description.output_sampling_frequency),
        wave="sine",
    )

    # example = dataset[args.index]
    # spectrograms = wavesim_to_batvision_spectrogram(
    #     (subset_recordings_dict(example, receiver_indices)).to(get_compute_device())
    # ).cpu()
    # plt.imshow(spectrograms[0].flip(0), cmap="gray")
    # plt.show()
    # exit(-1)

    recordings = dataset[args.index][k_sensor_recordings][receiver_indices]
    recordings_convolved = convolve_recordings(chirp, recordings).squeeze(0)

    fig, axes = plt.subplots(2, 1, figsize=(8, 4), dpi=80)

    axes[0].title.set_text("Impulse Response (Time Domain)")
    for i in range(recordings.shape[0]):
        axes[0].plot(recordings[i] + 0.003 * i, c=(0, 0, 0, 1.0), linewidth=2)

    # axes[1].title.set_text("Impulse Response (Frequency Domain)")
    # axes[1].plot(
    #     fft.rfftfreq(
    #         description.output_length, d=(1.0 / description.output_sampling_frequency)
    #     ),
    #     np.abs(fft.rfft(first_recording.numpy())),
    # )

    # axes[2].title.set_text("FM Chirp (Time Domain)")
    # axes[2].plot(chirp)

    axes[1].title.set_text("Combined (Time Domain)")
    for i in range(recordings_convolved.shape[0]):
        axes[1].plot(
            recordings_convolved[i, :1024] + 0.0006 * i, c=(0, 0, 0, 1.0), linewidth=2
        )

    # axes[4].title.set_text("Combined (Frequency Domain)")
    # axes[4].plot(
    #     fft.rfftfreq(
    #         description.output_length, d=(1.0 / description.output_sampling_frequency)
    #     ),
    #     np.abs(fft.rfft(first_recording_convolved.numpy())),
    # )

    axes[1].set_xlim(0, description.output_length - 1)
    axes[1].set_xlim(0, 1023)  # description.output_length - 1)
    # axes[3].set_xlim(0, description.output_length - 1)

    plt.show()


if __name__ == "__main__":
    main()
