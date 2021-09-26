import fix_dead_command_line

import matplotlib.pyplot as plt
import torch
import math
import numpy as np
import scipy.fft as fft
from argparse import ArgumentParser

from current_simulation_description import make_simulation_description
from dataset3d import WaveDataset3d, k_sensor_recordings
from which_device import get_compute_device
from signals_and_geometry import make_fm_chirp, convolve_recordings


def main():
    parser = ArgumentParser()
    parser.add_argument("path_to_dataset", type=str)
    parser.add_argument("index", nargs="?", type=int, default=0)
    parser.add_argument(
        "--f0", type=float, dest="f0", help="chirp start frequency (Hz)", default=0.0
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

    first_recording = dataset[args.index][k_sensor_recordings][0]

    chirp = make_fm_chirp(
        begin_frequency_Hz=args.f0,
        end_frequency_Hz=args.f1,
        sampling_frequency=description.output_sampling_frequency,
        chirp_length_samples=math.ceil(args.l * description.output_sampling_frequency),
        wave="sine",
    )

    first_recording_convolved = convolve_recordings(
        chirp, first_recording.unsqueeze(0)
    ).squeeze(0)

    fig, axes = plt.subplots(5, 1, figsize=(8, 4), dpi=80)

    axes[0].title.set_text("Impulse Response (Time Domain)")
    axes[0].plot(first_recording)

    axes[1].title.set_text("Impulse Response (Frequency Domain)")
    axes[1].plot(
        fft.rfftfreq(
            description.output_length, d=(1.0 / description.output_sampling_frequency)
        ),
        np.abs(fft.rfft(first_recording.numpy())),
    )

    axes[2].title.set_text("FM Chirp (Time Domain)")
    axes[2].plot(chirp)

    axes[3].title.set_text("Combined (Time Domain)")
    axes[3].plot(first_recording_convolved)

    axes[4].title.set_text("Combined (Frequency Domain)")
    axes[4].plot(
        fft.rfftfreq(
            description.output_length, d=(1.0 / description.output_sampling_frequency)
        ),
        np.abs(fft.rfft(first_recording_convolved.numpy())),
    )

    axes[0].set_xlim(0, description.output_length - 1)
    axes[2].set_xlim(0, description.output_length - 1)
    axes[3].set_xlim(0, description.output_length - 1)

    plt.show()


if __name__ == "__main__":
    main()
