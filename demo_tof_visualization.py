import fix_dead_command_line

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import numpy as np

from argparse import ArgumentParser
from current_simulation_description import make_simulation_description
from tof_utils import (
    SplitSize,
    make_receiver_indices,
    render_slices_prediction,
    render_slices_ground_truth,
    split_till_it_fits,
    time_of_flight_crop,
    colourize_sdf,
)
from utils import assert_eq, is_power_of_2
from dataset3d import WaveDataset3d
from the_device import the_device


class SimpleTOFPredictor(nn.Module):
    def __init__(
        self,
        speed_of_sound,
        sampling_frequency,
        recording_length_samples,
        crop_length_samples,
        emitter_location,
        receiver_locations,
        canonical_echo,
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

        assert isinstance(canonical_echo, torch.Tensor)
        assert_eq(canonical_echo.shape, (crop_length_samples,))
        self.canonical_echo = canonical_echo

    def forward(self, recordings, sample_locations):
        recordings_cropped = time_of_flight_crop(
            recordings=recordings,
            sample_locations=sample_locations,
            emitter_location=self.emitter_location,
            receiver_locations=self.receiver_locations,
            speed_of_sound=self.speed_of_sound,
            sampling_frequency=self.sampling_frequency,
            crop_length_samples=self.crop_length_samples,
        )

        B1, B2, R, L = recordings_cropped.shape

        # magnitude = torch.sum(torch.sum(torch.square(recordings_cropped), dim=2), dim=2)

        # magnitude = torch.sum(torch.square(torch.sum(recordings_cropped, dim=2)), dim=2)

        # magnitude = torch.sum(torch.sum(recordings_cropped, dim=2), dim=2)

        # magnitude = torch.sum(
        #     torch.sum(
        #         (recordings_cropped * self.canonical_echo.reshape(1, 1, 1, L)), dim=3
        #     ),
        #     dim=2,
        # )

        products = torch.sum(
            recordings_cropped * self.canonical_echo.reshape(1, 1, 1, L),
            dim=3,
        )

        products = torch.clamp(products, min=0.0)

        # threshold = 1e-5  # 0.0001
        # products[products < threshold] = 0.0

        magnitude = torch.sum(products, dim=2)

        assert_eq(magnitude.shape, (B1, B2))

        return magnitude


def colourize_bw_log(x):
    assert isinstance(x, torch.Tensor)
    assert len(x.shape) == 2
    xmin = torch.min(x).item()
    xmax = torch.max(x).item()
    print(f"Note: the min is {xmin} and the max is {xmax}")
    vmin = 1e-5
    vmax = 1e-3
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
    parser.add_argument("--tofcropsize", type=int, dest="tofcropsize", default=128)
    parser.add_argument("--nx", type=int, dest="nx", default=2)
    parser.add_argument("--ny", type=int, dest="ny", default=2)
    parser.add_argument("--nz", type=int, dest="nz", default=2)
    args = parser.parse_args()

    description = make_simulation_description()
    dataset = WaveDataset3d(description, "dataset_train.h5")

    sensor_indices = make_receiver_indices(
        args.nx,
        args.ny,
        args.nz,
    )

    # for i in range(49, 1000):
    for i in [7, 9, 24, 27, 29, 36, 43, 46, 53, 61]:
        print(i)
        # 7 - single small circle, kinda far away
        # 9 - single small circle, far away
        # 24 - single small circle, close
        # 27 - single large circle, far away
        # 29 - single large circle, kinda far away
        # 36 - single large circle, medium distance
        # 43 - two small circles, medium and kinda far
        # 46 - single large circle, kinda close
        # 53 - lotta stuff going on
        # 61 - near and far circles
        example = dataset[i]

        # TODO: add fm chirp?

        recordings = example["sensor_recordings"][sensor_indices].to(the_device)

        canonical_echo = recordings[0, : args.tofcropsize]

        obstacles = example["sdf"].to(the_device)

        model = SimpleTOFPredictor(
            speed_of_sound=description.air_properties.speed_of_sound,
            sampling_frequency=description.output_sampling_frequency,
            recording_length_samples=description.output_length,
            crop_length_samples=args.tofcropsize,
            emitter_location=description.emitter_location,
            receiver_locations=description.sensor_locations[sensor_indices],
            canonical_echo=canonical_echo,
        ).to(the_device)

        splits = SplitSize("render_slices_prediction")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=80)

        axes[0].plot(canonical_echo.cpu())

        axes[1].imshow(
            render_slices_ground_truth(
                obstacle_map=obstacles,
                description=description,
                colour_function=colourize_sdf,
            ).permute(1, 2, 0)
        )

        axes[2].imshow(
            split_till_it_fits(
                render_slices_prediction,
                splits,
                model=model,
                recordings=recordings,
                description=description,
                colour_function=colourize_bw_log,
            ).permute(1, 2, 0)
        )

        plt.show()


if __name__ == "__main__":
    main()
