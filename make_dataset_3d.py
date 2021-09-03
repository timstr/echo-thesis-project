import fix_dead_command_line
import cleanup_when_killed

import h5py
import numpy as np
import torch
import torch.nn as nn
import os
import math
from argparse import ArgumentParser

from simulation_description import SimulationDescription
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
from assert_eq import assert_eq
from kwave_util import make_ball
from signals_and_geometry import obstacle_map_to_sdf


def make_inner_outer_partitions():
    f = np.cbrt(0.5)

    f_lo = 0.5 - 0.5 * f
    f_hi = 0.5 + 0.5 * f

    x_min = Nx - minimum_x_units
    y_min = 0
    z_min = 0
    x_extent = Nx - minimum_x_units
    y_extent = Ny
    z_extent = Nz

    x_lo = x_min + round(x_extent * f_lo)
    x_hi = x_min + round(x_extent * f_hi)
    y_lo = y_min + round(y_extent * f_lo)
    y_hi = y_min + round(y_extent * f_hi)
    z_lo = z_min + round(z_extent * f_lo)
    z_hi = z_min + round(z_extent * f_hi)

    assert (
        abs(
            (
                ((x_hi - x_lo) * (y_hi - y_lo) * (z_hi - z_lo))
                / ((x_extent * y_extent * z_extent))
            )
            - 0.5
        )
        < 0.02  # NOTE: I know this error margin is a bit lousy, but it's the best I can do right now.
    )

    return (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)


def middle_is_empty(obstacles):
    if isinstance(obstacles, np.ndarray):
        obstacles = torch.tensor(obstacles)
    assert isinstance(obstacles, torch.Tensor)
    assert_eq(obstacles.shape, (Nx, Ny, Nz))
    assert_eq(obstacles.dtype, torch.bool)
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = make_inner_outer_partitions()
    return torch.all(obstacles[x_lo:x_hi, y_lo:y_hi, z_lo:z_hi] == 0)


def outside_is_empty(obstacles):
    if isinstance(obstacles, np.ndarray):
        obstacles = torch.tensor(obstacles)
    assert isinstance(obstacles, torch.Tensor)
    assert_eq(obstacles.shape, (Nx, Ny, Nz))
    assert_eq(obstacles.dtype, torch.bool)

    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = make_inner_outer_partitions()

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
        torch.all(obstacles[:x_lo, :, :] == 0)  # front
        and torch.all(obstacles[x_hi:, :, :] == 0)  # back
        and torch.all(obstacles[x_lo:x_hi, :y_lo, :] == 0)  # middle bottom
        and torch.all(obstacles[x_lo:x_hi, y_hi:, :] == 0)  # middle top
        and torch.all(obstacles[x_lo:x_hi, y_lo:y_hi, :z_lo] == 0)  # middle left
        and torch.all(obstacles[x_lo:x_hi, y_lo:y_hi, z_hi:] == 0)  # middle right
    )


def resample_echo4ch_obstacles(echo4ch_obstacles, description):
    echo4ch_grid_size = 64  # grid units
    echo4ch_spatial_extent = 0.64  # m
    assert isinstance(echo4ch_obstacles, np.ndarray)
    assert_eq(echo4ch_obstacles.shape, (64, 64, 64))
    assert isinstance(description, SimulationDescription)

    # HACK
    # echo4ch_obstacles[...] = 1.0

    sim_x_grid_size = Nx - minimum_x_units  # grid units
    sim_y_grid_size = Ny  # grid units
    sim_z_grid_size = Nz  # grid units
    sim_x_spatial_extent = sim_x_grid_size * description.dx  # m
    sim_y_spatial_extent = sim_y_grid_size * description.dy  # m
    sim_z_spatial_extent = sim_z_grid_size * description.dz  # m
    sim_to_echo4ch_x_ratio = sim_x_spatial_extent / echo4ch_spatial_extent
    sim_to_echo4ch_y_ratio = sim_y_spatial_extent / echo4ch_spatial_extent
    sim_to_echo4ch_z_ratio = sim_z_spatial_extent / echo4ch_spatial_extent
    sim_to_echo4ch_x_min_index = -sim_to_echo4ch_x_ratio
    sim_to_echo4ch_y_min_index = -sim_to_echo4ch_y_ratio
    sim_to_echo4ch_z_min_index = -sim_to_echo4ch_z_ratio
    sim_to_echo4ch_x_max_index = +sim_to_echo4ch_x_ratio
    sim_to_echo4ch_y_max_index = +sim_to_echo4ch_y_ratio
    sim_to_echo4ch_z_max_index = +sim_to_echo4ch_z_ratio
    ls_x = torch.linspace(
        sim_to_echo4ch_x_min_index, sim_to_echo4ch_x_max_index, sim_x_grid_size
    )
    ls_y = torch.linspace(
        sim_to_echo4ch_y_min_index, sim_to_echo4ch_y_max_index, sim_y_grid_size
    )
    ls_z = torch.linspace(
        sim_to_echo4ch_z_min_index, sim_to_echo4ch_z_max_index, sim_z_grid_size
    )

    gx, gy, gz = torch.meshgrid([ls_x, ls_y, ls_z])
    assert_eq(
        gx.shape,
        (
            sim_x_grid_size,
            sim_y_grid_size,
            sim_z_grid_size,
        ),
    )
    assert_eq(
        gy.shape,
        (
            sim_x_grid_size,
            sim_y_grid_size,
            sim_z_grid_size,
        ),
    )
    assert_eq(
        gz.shape,
        (
            sim_x_grid_size,
            sim_y_grid_size,
            sim_z_grid_size,
        ),
    )
    all_locations = torch.stack([gx, gy, gz], dim=-1)
    assert_eq(
        all_locations.shape, (sim_x_grid_size, sim_y_grid_size, sim_z_grid_size, 3)
    )

    # for grid_sample: batch, features, output depth, output height, output width, xyz coordinates
    t_grid = all_locations.reshape(
        1,  # N
        sim_x_grid_size,  # D_out
        sim_y_grid_size,  # H_out
        sim_z_grid_size,  # W_out
        3,  # yep, just 3
    )

    input_tensor = (
        torch.tensor(echo4ch_obstacles)
        .permute(1, 2, 0)
        .reshape(
            1,  # N
            1,  # C
            echo4ch_grid_size,  # D_in
            echo4ch_grid_size,  # H_in
            echo4ch_grid_size,  # W_in
        )
        .float()
    )

    values = nn.functional.grid_sample(
        input=input_tensor,
        grid=t_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    assert_eq(values.shape, (1, 1, sim_x_grid_size, sim_y_grid_size, sim_z_grid_size))

    values = values.reshape(sim_x_grid_size, sim_y_grid_size, sim_z_grid_size)

    return values > 0.0


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
    parser.add_argument("--verbose", dest="verbose", default=False, action="store_true")
    parser.add_argument("--append", dest="append", default=False, action="store_true")
    args = parser.parse_args()
    numworkers = args.numworkers
    count = args.count
    workerindex = args.workerindex
    mode = args.mode
    append = args.append

    assert numworkers >= 1
    assert workerindex >= 0 and workerindex < numworkers

    desc = make_simulation_description()

    desc.print_summary()

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
                attempts = 0
                while True:
                    attempts += 1
                    obs = make_random_obstacles(desc)
                    if filter(obs):
                        break
                if args.verbose:
                    print(
                        f"Found valid random obstacles after {attempts} attempt{'' if attempts == 1 else 's'}"
                    )
                yield obs, mode

    elif mode == mode_orbiting_sphere:
        radius = math.ceil(0.05 / desc.dz)
        margin = 4
        min_x = minimum_x_units + margin + radius
        min_y = margin + radius
        min_z = margin + radius
        max_x = desc.Nx - margin - radius
        max_y = desc.Ny - margin - radius
        max_z = desc.Nz - margin - radius

        def obstacle_generator():
            i = 0
            while i < count:
                t = i / count
                print(t)
                tx = 0.5 - 0.5 * math.sin(2.0 * math.pi * t)
                ty = 0.5 - 0.5 * math.sin(4.0 * math.pi * t)
                tz = 0.5 + 0.5 * math.cos(6.0 * math.pi * t)
                ix = min_x + tx * (max_x - min_x)
                iy = min_y + ty * (max_y - min_y)
                iz = min_z + tz * (max_z - min_z)
                obs = make_ball(desc.Nx, desc.Ny, desc.Nz, ix, iy, iz, radius)
                yield obs, f"orbiting-sphere {i}/{count}"
                i += 1

        # HACK
        for i, f in enumerate(obstacle_generator()):
            pass
        exit(0)

    elif mode == mode_echo4ch:
        echo4ch_obstacle_path = os.environ.get("ECHO4CH_OBSTACLES")

        if echo4ch_obstacle_path is None:
            raise Exception(
                "Please set the ECHO4CH_OBSTACLES environment variable to point to the ECHO4CH obstacles HDF5 file"
            )

        def obstacle_generator():
            with h5py.File(echo4ch_obstacle_path, "r") as obstacles_h5file:
                obstacles_ds = H5DS(
                    name="obstacles",
                    dtype=np.bool8,
                    shape=(64, 64, 64),
                    extensible=True,
                )
                assert obstacles_ds.exists(obstacles_h5file)
                i = 0
                obstacles_ds_size = obstacles_ds.count(obstacles_h5file)
                N = obstacles_ds_size
                if count is not None:
                    N = min(count, N)
                magic_prime_number = 7919
                assert (obstacles_ds_size % magic_prime_number) != 0
                while i < N:
                    i_permuted = (i * magic_prime_number) % obstacles_ds_size
                    echo4ch_obstacles = obstacles_ds.read(
                        obstacles_h5file, index=i_permuted
                    )
                    sim_roi_obstacles = resample_echo4ch_obstacles(
                        echo4ch_obstacles, desc
                    )
                    sim_obstacles = np.zeros(
                        (desc.Nx, desc.Ny, desc.Nz), dtype=np.bool8
                    )
                    sim_obstacles[minimum_x_units:, :, :] = sim_roi_obstacles
                    yield sim_obstacles, f"ECHO4CH obstacles {i} permuted to {i_permuted}"
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

    if os.path.exists(dataset_path) and not append:
        print(
            f"Error: attempted to create a dataset file at {dataset_path} but it already exists. Please provide a different path for DATASET_OUTPUT or use the --append flag."
        )
        exit(-1)

    with WaveDataset3d(desc, dataset_path, write=True) as dataset:
        for i, (o, s) in enumerate(obstacles_subset()):
            print(f'{i} - Creating dataset example "{s}"')
            if dataset.contains(o):
                print(
                    "Warning: duplicate obstacles found, this example is being skipped"
                )
                continue
            desc.set_obstacles(o)
            results = desc.run(verbose=args.verbose)
            sdf = obstacle_map_to_sdf(torch.tensor(o).cuda(), desc).cpu().numpy()
            dataset.append_to_dataset(obstacles=o, recordings=results, sdf=sdf)


if __name__ == "__main__":
    main()
