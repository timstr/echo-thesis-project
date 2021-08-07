from tof_utils import obstacle_map_to_sdf
import fix_dead_command_line

import h5py
import numpy as np
import torch
import os
from argparse import ArgumentParser

from h5ds import H5DS
from dataset3d import WaveDataset3d
from current_simulation_description import (
    make_random_obstacles,
    make_simulation_description,
    minimum_x_units,
    Nx,
    Ny,
    Nz,
)
from utils import assert_eq


def middle_is_empty(obstacles):
    if isinstance(obstacles, np.ndarray):
        obstacles = torch.tensor(obstacles)
    assert isinstance(obstacles, torch.Tensor)
    assert_eq(obstacles.shape, (Nx, Ny, Nz))
    assert_eq(obstacles.dtype, torch.bool)
    xlo = minimum_x_units + ((Nx - minimum_x_units) // 4)
    xhi = minimum_x_units + ((Nx - minimum_x_units) * 3 // 4)
    ylo = Ny // 4
    yhi = Ny * 3 // 4
    zlo = Nz // 4
    zhi = Nz * 3 // 4
    return torch.all(obstacles[xlo:xhi, ylo:yhi, zlo:zhi] == 0)


def outside_is_empty(obstacles):
    if isinstance(obstacles, np.ndarray):
        obstacles = torch.tensor(obstacles)
    assert isinstance(obstacles, torch.Tensor)
    assert_eq(obstacles.shape, (Nx, Ny, Nz))
    assert_eq(obstacles.dtype, torch.bool)
    xlo = minimum_x_units + ((Nx - minimum_x_units) // 4)
    xhi = minimum_x_units + ((Nx - minimum_x_units) * 3 // 4)
    ylo = Ny // 4
    yhi = Ny * 3 // 4
    zlo = Nz // 4
    zhi = Nz * 3 // 4
    #  front
    # ########
    # ########
    # ########
    # ########
    #
    #  middle
    # ######## <- middle top
    # ##    ## <- middle left and right
    # ######## <- middle bottom
    #
    #  back
    # ########
    # ########
    # ########
    # ########

    return (
        torch.all(obstacles[:xlo, :, :] == 0)  # front
        and torch.all(obstacles[xhi:, :, :] == 0)  # back
        and torch.all(obstacles[xlo:xhi, :ylo, :] == 0)  # middle bottom
        and torch.all(obstacles[xlo:xhi, yhi:, :] == 0)  # middle top
        and torch.all(obstacles[xlo:xhi, ylo:yhi, :zlo] == 0)  # middle left
        and torch.all(obstacles[xlo:xhi, ylo:yhi, zhi:] == 0)  # middle right
    )


def main():
    mode_random = "random"
    mode_random_outer = "random-outer"
    mode_random_inner = "random-inner"
    mode_orbiting_sphere = "orbiting-sphere"
    mode_echo4ch = "echo4ch"

    parser = ArgumentParser()
    parser.add_argument("--numworkers", type=int, dest="numworkers", required=True)
    parser.add_argument("--count", type=int, dest="count", required=False)
    parser.add_argument("--workerindex", type=int, dest="workerindex", required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            mode_random,
            mode_random_inner,
            mode_random_outer,
            mode_orbiting_sphere,
            mode_echo4ch,
        ],
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

    if mode in [mode_random, mode_random_inner, mode_random_outer]:
        if count is None:
            raise Exception(
                "Please specify the --count argument when using random mode"
            )

        def filter(obs):
            if mode == mode_random_inner:
                return outside_is_empty(obs)
            elif mode == mode_random_outer:
                return middle_is_empty(obs)
            return True

        def obstacle_generator():
            for i in range(count):
                while True:
                    obs = make_random_obstacles(desc)
                    if filter(obs):
                        break
                yield obs, mode

    elif mode == mode_orbiting_sphere:
        raise Exception("orbiting-sphere mode is not implemented")
    elif mode == mode_echo4ch:
        echo4ch_obstacle_path = os.environ.get("ECHO4CH_OBSTACLES")

        if echo4ch_obstacle_path is None:
            raise Exception(
                "Please set the ECHO4CH_OBSTACLES environment variable to point to the ECHO4CH obstacles HDF5 file"
            )

        def obstacle_generator():
            with h5py.file(echo4ch_obstacle_path, "r") as obstacles_h5file:
                obstacles_ds = H5DS(
                    name="obstacles",
                    dtype=np.bool8,
                    shape=(64, 64, 64),
                    extensible=True,
                )
                assert obstacles_ds.exists(obstacles_h5file)
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

    if os.path.exists(dataset_path):
        print(
            f"Error: attempted to create a dataset file at {dataset_path} but it already exists"
        )
        exit(-1)

    for i, (o, s) in enumerate(obstacles_subset()):
        print(f'{i} - Creating dataset example "{s}"')
        desc.set_obstacles(o)
        results = desc.run()
        sdf = obstacle_map_to_sdf(torch.tensor(o).cuda(), desc).cpu().numpy()
        with WaveDataset3d(desc, dataset_path, write=True) as dataset:
            dataset.append_to_dataset(obstacles=o, recordsings=results, sdf=sdf)


if __name__ == "__main__":
    main()
