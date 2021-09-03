import fix_dead_command_line

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import numpy as np
from argparse import ArgumentParser

from current_simulation_description import (
    make_receiver_indices,
    make_simulation_description,
)
from assert_eq import assert_eq
from utils import is_power_of_2
from dataset3d import WaveDataset3d, k_sensor_recordings, k_sdf
from the_device import the_device
from split_till_it_fits import SplitSize, split_till_it_fits
from visualization import (
    colourize_sdf,
    render_slices_ground_truth,
    render_slices_prediction,
)
from signals_and_geometry import convolve_recordings, make_fm_chirp, time_of_flight_crop


class SimpleTOFPredictor(nn.Module):
    def __init__(
        self,
        speed_of_sound,
        sampling_frequency,
        recording_length_samples,
        crop_length_samples,
        emitter_location,
        receiver_locations,
    ):
        super(SimpleTOFPredictor, self).__init__()

        assert isinstance(speed_of_sound, float)
        self.speed_of_sound = speed_of_sound

        assert isinstance(sampling_frequency, float)
        self.sampling_frequency = sampling_frequency

        assert isinstance(recording_length_samples, int)
        assert is_power_of_2(recording_length_samples)
        self.recording_length_samples = recording_length_samples

        assert isinstance(crop_length_samples, int)
        assert is_power_of_2(crop_length_samples)
        self.crop_length_samples = crop_length_samples

        assert isinstance(emitter_location, np.ndarray)
        assert_eq(emitter_location.shape, (3,))
        assert_eq(emitter_location.dtype, np.float32)
        self.emitter_location = nn.parameter.Parameter(
            data=torch.tensor(emitter_location, dtype=torch.float32),
            requires_grad=False,
        )

        assert isinstance(receiver_locations, np.ndarray)
        assert receiver_locations.dtype == np.float32
        assert receiver_locations.shape[1:] == (3,)
        num_receivers = receiver_locations.shape[0]
        self.num_receivers = num_receivers
        receiver_locations_tensor = torch.tensor(
            receiver_locations, dtype=torch.float32
        )
        assert receiver_locations_tensor.shape == (num_receivers, 3)
        self.receiver_locations = nn.parameter.Parameter(
            data=receiver_locations_tensor,
            requires_grad=False,
        )

        self.window_fn = nn.parameter.Parameter(
            data=(
                0.5 - 0.5 * torch.cos(torch.linspace(0.0, math.pi, crop_length_samples))
            ).reshape(1, 1, 1, crop_length_samples),
            requires_grad=False,
        )

    def forward(self, recordings, sample_locations):
        recordings_cropped = time_of_flight_crop(
            recordings=recordings,
            sample_locations=sample_locations,
            emitter_location=self.emitter_location,
            receiver_locations=self.receiver_locations,
            speed_of_sound=self.speed_of_sound,
            sampling_frequency=self.sampling_frequency,
            crop_length_samples=self.crop_length_samples,
            # apply_amplitude_correction=True,
        )

        # recordings_cropped = sclog(recordings_cropped)

        B1, B2, R, L = recordings_cropped.shape

        recordings_windowed = recordings_cropped * self.window_fn

        # magnitude = torch.sum(
        #     torch.sum(recordings_windowed, dim=2),
        #     dim=2,
        # )

        magnitude = torch.mean(
            torch.mean(torch.square(recordings_windowed), dim=2), dim=2
        ) / (torch.mean(torch.var(recordings_windowed, dim=2), dim=2) + 1e-3)

        # magnitude = torch.sum(torch.square(torch.sum(recordings_cropped, dim=2)), dim=2)

        # magnitude = torch.sum(torch.sum(recordings_cropped, dim=2), dim=2)

        # magnitude = torch.sum(
        #     torch.sum(
        #         (recordings_cropped * self.canonical_echo.reshape(1, 1, 1, L)), dim=3
        #     ),
        #     dim=2,
        # )

        # products = torch.sum(
        #     recordings_cropped * self.canonical_echo.reshape(1, 1, 1, L),
        #     dim=3,
        # )

        # products = torch.clamp(products, min=0.0)

        # threshold = 1e-5  # 0.0001
        # products[products < threshold] = 0.0

        # magnitude = torch.sum(products, dim=2)

        assert_eq(magnitude.shape, (B1, B2))

        return magnitude


def colourize_bw_log(x, vmin, vmax):
    assert isinstance(x, torch.Tensor)
    assert isinstance(vmin, float)
    assert isinstance(vmax, float)
    assert vmin > 0.0
    assert vmax > vmin
    assert len(x.shape) == 2
    xmin = torch.min(torch.abs(x)).item()
    xmax = torch.max(torch.abs(x)).item()
    print(f"Note: the abs min is {xmin} and the abs max is {xmax}")
    logmin = math.log(vmin)
    logmax = math.log(vmax)
    linlogposx = (torch.log(torch.clamp(x, min=vmin, max=vmax)) - logmin) / (
        logmax - logmin
    )
    if xmin >= 0.0:
        return linlogposx.unsqueeze(0).repeat(3, 1, 1)
    linlognegx = (torch.log(torch.clamp(-x, min=vmin, max=vmax)) - logmin) / (
        logmax - logmin
    )
    return torch.stack([linlogposx, torch.zeros_like(linlogposx), linlognegx], dim=0)


def main():
    parser = ArgumentParser()
    parser.add_argument("path_to_dataset", type=str)
    parser.add_argument("index", nargs="?", type=int, default=0)
    parser.add_argument("--tofcropsize", type=int, dest="tofcropsize", default=64)
    parser.add_argument("--nx", type=int, dest="nx", default=4)
    parser.add_argument("--ny", type=int, dest="ny", default=4)
    parser.add_argument("--nz", type=int, dest="nz", default=4)
    parser.add_argument(
        "--f0", type=float, dest="f0", help="chirp start frequency (Hz)", default=0.0
    )
    parser.add_argument(
        "--f1", type=float, dest="f1", help="end frequency (Hz)", default=20_000.0
    )
    parser.add_argument(
        "--l", type=float, dest="l", help="chirp duration (seconds)", default=0.001
    )
    parser.add_argument(
        "--vmin", type=float, dest="vmin", help="minimum displayable value", default=0.1
    )
    parser.add_argument(
        "--vmax",
        type=float,
        dest="vmax",
        help="maximum displayable value",
        default=10.0,
    )
    args = parser.parse_args()

    description = make_simulation_description()
    dataset = WaveDataset3d(description, args.path_to_dataset)

    sensor_indices = make_receiver_indices(
        args.nx,
        args.ny,
        args.nz,
    )

    chirp = torch.tensor(
        make_fm_chirp(
            begin_frequency_Hz=args.f0,
            end_frequency_Hz=args.f1,
            sampling_frequency=description.output_sampling_frequency,
            chirp_length_samples=math.ceil(
                args.l * description.output_sampling_frequency
            ),
            wave="sine",
        )
    ).float()

    chirp = chirp.to(the_device)

    splits = SplitSize("render_slices_prediction")

    if args.index < 0 or args.index >= len(dataset):
        print(
            f"The dataset index {args.index} is out of bounds. Valid indices are 0 to {len(dataset) - 1}"
        )
    example = dataset[args.index]

    recordings_ir = example[k_sensor_recordings][sensor_indices].to(the_device)

    recordings_chirp = convolve_recordings(chirp, recordings_ir, description)
    # recordings_chirp = recordings_ir

    obstacles = example[k_sdf].to(the_device)

    model = SimpleTOFPredictor(
        speed_of_sound=description.air_properties.speed_of_sound,
        sampling_frequency=description.output_sampling_frequency,
        recording_length_samples=description.output_length,
        crop_length_samples=args.tofcropsize,
        emitter_location=description.emitter_location,
        receiver_locations=description.sensor_locations[sensor_indices],
    ).to(the_device)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=80)

    axes[0].imshow(
        render_slices_ground_truth(
            obstacle_map=obstacles,
            description=description,
            colour_function=colourize_sdf,
        ).permute(1, 2, 0)
    )

    axes[1].imshow(
        split_till_it_fits(
            render_slices_prediction,
            splits,
            model=model,
            recordings=recordings_chirp,
            description=description,
            colour_function=lambda x: colourize_bw_log(x, args.vmin, args.vmax),
        ).permute(1, 2, 0)
    )

    plt.show()


if __name__ == "__main__":
    main()
