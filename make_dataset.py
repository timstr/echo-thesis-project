import fix_dead_command_line

import os
import numpy as np
from shape_types import CIRCLE, RECTANGLE
from featurize import all_possible_obstacles, obstacles_occluded
import pickle
import torch
import math

# import matplotlib.pyplot as plt

from argparse import ArgumentParser

from the_device import the_device
from wave_field import Field
from wavesim_params import (
    wavesim_field_size,
    wavesim_duration,
)
from config_constants import (
    wavesim_emitter_locations,
    wavesim_receiver_locations,
)
from utils import progress_bar


def main():
    parser = ArgumentParser()
    parser.add_argument("--offset", type=int, dest="offset", required=True)
    parser.add_argument("--end", type=int, dest="end", required=True)
    parser.add_argument("--numworkers", type=int, dest="numworkers", required=True)
    parser.add_argument("--workerindex", type=int, dest="workerindex", required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["normal", "orbiting-circle", "widening-rectangle"],
        dest="mode",
        required=True,
    )
    args = parser.parse_args()

    emitter_radius = 2
    num_emitters = len(wavesim_emitter_locations)

    receiver_locations = torch.tensor(wavesim_receiver_locations, dtype=torch.float)
    num_receivers = len(receiver_locations)
    receiver_indices_chunked = (
        torch.tensor(wavesim_receiver_locations, dtype=torch.long)
        .t()
        .chunk(chunks=num_receivers, dim=0)
    )

    if args.mode == "normal":
        all_configurations = list(
            all_possible_obstacles(
                max_num_obstacles=2,
                min_dist=0.05,
                min_size=0.1,
                max_size=0.3,
                top=0.1,
                bottom=0.8,
                left=0.1,
                right=0.9,
                num_size_increments=3,
                num_space_increments=4,
                num_angle_increments=4,
            )
        )
    elif args.mode == "orbiting-circle":
        num_steps = 1024
        rad = 0.1
        xmin = rad + 0.25
        xmax = 0.75 - rad
        ymin = rad + 0.1
        ymax = 0.6

        def yx(t):
            assert t >= 0.0 and t <= 1.0
            s = 4 * t
            s -= math.floor(s)
            if t < 0.25:  # top edge, moving right
                return ymin, (xmin + s * (xmax - xmin))
            elif t < 0.5:  # right edge, moving down
                return (ymin + s * (ymax - ymin)), xmax
            elif t < 0.75:  # bottom edge, moving left
                return ymax, (xmax + s * (xmin - xmax))
            else:  # left edge, moving up
                return (ymax + s * (ymin - ymax)), xmin

        def make_circle(t):
            y, x = yx(t)
            return [(CIRCLE, y, x, rad)]

        all_configurations = [
            make_circle(t) for t in np.linspace(0.0, 1.0, num_steps, endpoint=False)
        ]
    elif args.mode == "widening-rectangle":
        num_steps = 64
        height = 0.05
        topwidth = 0.25
        bottomwidthmin = 0.1
        bottomwidthmax = 0.5

        def make_rectangles(t):
            bottomwidth = bottomwidthmin + t * (bottomwidthmax - bottomwidthmin)
            return [
                (RECTANGLE, 0.25, 0.5, height, topwidth, 0.0),
                (RECTANGLE, 0.50, 0.5, height, bottomwidth, 0.0),
            ]

        all_configurations = [
            make_rectangles(t) for t in np.linspace(0.0, 1.0, num_steps)
        ]
    else:
        raise Exception("Unrecognized dataset mode")
    dataset_size = len(all_configurations)

    # returns list_of_obstacles, sound_buffer
    def make_example(example_index):
        with torch.no_grad():
            print("Creating field")
            field = Field(wavesim_field_size)
            obstacles = all_configurations[example_index]
            field.add_obstacles(obstacles)
            print("Simulating waves")
            receiver_buf = torch.zeros(
                num_emitters, num_receivers, wavesim_duration, dtype=torch.float32
            ).to(the_device)
            for i_emitter, (emitter_y, emitter_x) in enumerate(
                wavesim_emitter_locations
            ):
                field.silence()
                field.get_field()[
                    emitter_y - emitter_radius : emitter_y + emitter_radius,
                    emitter_x - emitter_radius : emitter_x + emitter_radius,
                ] = 1.0
                for s in range(wavesim_duration):
                    field.step()
                    receiver_buf[i_emitter, :, s] = field.get_field()[
                        receiver_indices_chunked
                    ]
                progress_bar(i_emitter, num_emitters)

            max_amp = torch.max(torch.abs(receiver_buf)).item()
            if max_amp > 1e-6:
                receiver_buf *= 0.5 / max_amp
            sound_buf = receiver_buf.detach().cpu().numpy()
            occlusion = obstacles_occluded(obstacles)
            print(" Done")
            return {
                "obstacles": field.get_obstacles(),
                "impulse_responses": sound_buf,
                "occlusion": occlusion,
            }

    output_path = os.environ.get("OUTPUT_PATH")

    if output_path is None or not os.path.exists(output_path):
        raise Exception(
            "Please set the OUTPUT_PATH environment variable to point to the desired output directory"
        )

    num_digits = int(math.ceil(math.log10(dataset_size)))

    from_here = args.offset + args.workerindex
    to_there = min(dataset_size, args.end)
    skipping = args.numworkers
    for i in range(from_here, to_there, skipping):
        print("Creating example ", i)
        example = make_example(i)
        fname = f"example {str(i).zfill(num_digits)}.pkl"
        path = os.path.join(output_path, fname)
        with open(path, "wb") as outfile:
            pickle.dump(example, outfile)


if __name__ == "__main__":
    main()
