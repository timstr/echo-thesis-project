import fix_dead_command_line

import h5py
import numpy as np
import os
from argparse import ArgumentParser

from h5ds import H5DS
from dataset3d import WaveDataset3d
from current_simulation_description import (
    make_random_obstacles,
    make_simulation_description,
)


def main():
    mode_random = "random"
    mode_orbiting_sphere = "orbiting-sphere"
    mode_echo4ch = "echo4ch"

    parser = ArgumentParser()
    parser.add_argument("--numworkers", type=int, dest="numworkers", required=True)
    parser.add_argument("--count", type=int, dest="count", required=False)
    parser.add_argument("--workerindex", type=int, dest="workerindex", required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["random", "orbiting-circle", "echo4ch"],
        dest="mode",
        required=True,
    )
    args = parser.parse_args()
    numworkers = args.numworkers
    count = args.count
    workerindex = args.workerindex
    mode = args.mode

    assert numworkers >= 1
    assert workerindex >= 0 and workerindex < numworkers

    desc = make_simulation_description()

    dataset_path = os.environ.get("DATASET_OUTPUT")

    if dataset_path is None:
        raise Exception(
            "Please set the DATASET_OUTPUT environment variable to point to the HDF5 dataset file that should be created"
        )

    if mode == mode_random:
        if count is None:
            raise Exception(
                "Please specify the --count argument when using random mode"
            )

        def obstacle_generator():
            for i in range(count):
                yield make_random_obstacles(desc), "random obstacles"

    elif mode == mode_orbiting_sphere:
        raise Exception("orbiting-sphere mode is not implemented")
    elif mode == mode_echo4ch:
        echo4ch_obstacle_path = os.environ.get("ECHO4CH_OBSTACLES")

        if echo4ch_obstacle_path is None:
            raise Exception(
                "Please set the ECHO4CH_OBSTACLES environment variable to point to the ECHO4CH obstacles HDF5 file"
            )

        obstacles_h5file = h5py.File(echo4ch_obstacle_path, "r")
        obstacle_ds = H5DS(
            name="obstacles", dtype=np.bool8, shape=(64, 64, 64), extensible=True
        )
        assert obstacle_ds.exists(obstacles_h5file)

        def obstacle_generator():
            i = 0
            N = obstacle_ds.count(obstacles_h5file)
            if count is not None:
                N = min(count, N)
            while i < N:
                e = obstacle_ds.read(obstacles_h5file, index=i)
                o = np.zeros((desc.Nx, desc.Ny, desc.Nz), dtype=np.bool8)
                o[-64:, :, :] = e[:, 2:-2, 2:-2]
                yield o, f"ECHO4CH obstacle {i}"
                i += 1

    else:
        raise Exception(f'Unrecognized dataset mode: "{mode}"')

    def obstacles_subset():
        try:
            obstacles_all = obstacle_generator()
            for _ in range(workerindex):
                next(obstacles_all)
            while True:
                yield next(obstacles_all)
                for _ in range(numworkers - 1):
                    next(obstacles_all)
        except StopIteration:
            pass

    dataset = WaveDataset3d(desc, dataset_path, write=True)

    for i, (o, s) in enumerate(obstacles_subset()):
        print(f'{i} - Creating dataset example "{s}"')
        dataset.simulate_and_append_to_dataset(o)


if __name__ == "__main__":
    main()
