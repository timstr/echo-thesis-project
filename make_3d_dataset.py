import fix_dead_command_line

import datetime
from kwave_run_simulation import run_kwave_simulation
import math
import numpy as np
import h5py
import random

from kwave_simulation_description import AcousticMediumProperties, SimulationDescription
from kwave_util import (
    append_to_dataset,
    encode_str,
    make_ball,
    make_empty_extensible_dataset,
    read_scalar,
    write_array,
    write_scalar,
)

s_attribute_description = "description"
s_attribute_created_by = "created_by"
s_attribute_creation_date = "creation_date"

s_dataset_sensor_count = "sensor_count"
s_dataset_Nx = "Nx"
s_dataset_Ny = "Ny"
s_dataset_Nz = "Nz"
s_dataset_dx = "dx"
s_dataset_dy = "dy"
s_dataset_dz = "dz"
s_dataset_air_speed_of_sound = "air_speed_of_sound"
s_dataset_signal_sampling_frequency = "signal_sampling_frequency"
s_dataset_signal_length = "signal_length"

s_dataset_sensor_locations = "sensor_locations"

s_dataset_sensor_recordings = "sensor_recordings"
s_dataset_obstacles = "obstacles"


def init_dataset(h5_file, description):
    assert isinstance(h5_file, h5py.File)
    assert isinstance(description, SimulationDescription)
    # TODO
    # - add metadata from simulation description
    # - add empty, resizeable datasets for each example component:
    #     - time-series audio recording, as returned by run_kwave_simulation
    #     -
    h5_file.attrs[s_attribute_description] = "Echo dataset"
    h5_file.attrs[s_attribute_created_by] = "Tim Straubinger"
    h5_file.attrs[s_attribute_creation_date] = encode_str(
        datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
    )

    write_array(
        h5_file,
        s_dataset_sensor_locations,
        np.array(description.sensor_locations),
        dtype=np.int32,
    )
    write_scalar(
        h5_file, s_dataset_sensor_count, description.num_sensors, dtype=np.int32
    )

    write_scalar(h5_file, s_dataset_dx, description.dx, np.float32)
    write_scalar(h5_file, s_dataset_dy, description.dy, np.float32)
    write_scalar(h5_file, s_dataset_dz, description.dz, np.float32)

    write_scalar(h5_file, s_dataset_Nx, description.Nx, np.float32)
    write_scalar(h5_file, s_dataset_Ny, description.Ny, np.float32)
    write_scalar(h5_file, s_dataset_Nz, description.Nz, np.float32)

    write_scalar(
        h5_file,
        s_dataset_air_speed_of_sound,
        description.air_properties.speed_of_sound,
        np.float32,
    )
    write_scalar(
        h5_file,
        s_dataset_signal_sampling_frequency,
        description.output_sampling_frequency,
        np.float32,
    )
    write_scalar(
        h5_file, s_dataset_signal_length, description.output_length, np.float32
    )

    make_empty_extensible_dataset(
        h5_file,
        s_dataset_sensor_recordings,
        (description.num_sensors, description.output_length),
        dtype=np.float32,
    )
    make_empty_extensible_dataset(
        h5_file,
        s_dataset_obstacles,
        (description.Nx, description.Ny, description.Nz),
        dtype=np.bool8,
    )


def add_example_to_dataset(h5_file, description, simulation_results):
    assert isinstance(h5_file, h5py.File)
    assert isinstance(description, SimulationDescription)

    assert read_scalar(h5_file, s_dataset_sensor_count) == description.num_sensors
    assert read_scalar(h5_file, s_dataset_Nx) == description.Nx
    assert read_scalar(h5_file, s_dataset_Ny) == description.Ny
    assert read_scalar(h5_file, s_dataset_Nz) == description.Nz

    assert np.isclose(read_scalar(h5_file, s_dataset_dx), description.dx)
    assert np.isclose(read_scalar(h5_file, s_dataset_dy), description.dy)
    assert np.isclose(read_scalar(h5_file, s_dataset_dz), description.dz)
    assert np.isclose(
        read_scalar(h5_file, s_dataset_air_speed_of_sound),
        description.air_properties.speed_of_sound,
    )
    assert np.isclose(
        read_scalar(h5_file, s_dataset_signal_sampling_frequency),
        description.output_sampling_frequency,
    )
    assert np.isclose(
        read_scalar(h5_file, s_dataset_signal_length), description.output_length
    )

    assert simulation_results.shape == (
        description.num_sensors,
        description.output_length,
    )
    append_to_dataset(
        h5_file, s_dataset_sensor_recordings, simulation_results, dtype=np.float32
    )
    append_to_dataset(
        h5_file, s_dataset_obstacles, description.obstacle_mask, dtype=np.bool8
    )


c_air = 343.0
c_wood = 4000.0
c_human = 1540.0
c_dense_air = c_air * 2.0

rho_air = 1.225
rho_wood = 500.0
rho_human = 1010.0
rho_dense_air = rho_air * 2.0

Nx = 180
Ny = 60
Nz = 60

minimum_x_distance = 60

sensor_center_x = 16
sensor_center_y = Ny // 2
sensor_center_z = Nz // 2

sensor_count_x = 4
sensor_count_y = 4
sensor_count_z = 4

sensor_spacing = 4


def a_nice_description():

    sensor_extent_x = sensor_count_x * (sensor_spacing - 1)
    sensor_extent_y = sensor_count_y * (sensor_spacing - 1)
    sensor_extent_z = sensor_count_z * (sensor_spacing - 1)

    sensor_locations = []

    # record a small grid
    for i in range(sensor_count_x):
        for j in range(sensor_count_y):
            for k in range(sensor_count_z):
                x = round(sensor_center_x - (sensor_extent_x / 2) + sensor_spacing * i)
                y = round(sensor_center_y - (sensor_extent_y / 2) + sensor_spacing * j)
                z = round(sensor_center_z - (sensor_extent_z / 2) + sensor_spacing * k)
                sensor_locations.append((x, y, z))

    air_properties = AcousticMediumProperties(
        speed_of_sound=c_air,  # meters per second
        density=rho_air,  # kilograms per cubic meter
    )
    obstacle_properties = AcousticMediumProperties(
        speed_of_sound=c_dense_air,  # meters per second
        density=rho_dense_air,  # kilograms per cubic meter
        # speed_of_sound=c_wood,  # meters per second
        # density=rho_wood,  # kilograms per cubic meter
    )

    spatial_resolution = 1e-2  # meters
    Npml = 10  # spatial count
    # dt = 1e-7  # seconds
    dt = 1e-6  # seconds

    sampling_frequency = 96_000.0
    sampling_period = 1.0 / sampling_frequency

    wave_distance_per_time_step = air_properties.speed_of_sound * dt
    corner_to_corner_distance = (
        2.0 * math.sqrt(Nx ** 2 + Ny ** 2 + Nz ** 2) * spatial_resolution
    )
    Nt_original = math.ceil(corner_to_corner_distance / wave_distance_per_time_step)

    Nt_at_sampling_frequency = Nt_original * (dt / sampling_period)
    Nt_at_sampling_frequency_rounded = 2 ** (
        math.ceil(math.log2(Nt_at_sampling_frequency))
    )
    Nt = round(Nt_at_sampling_frequency_rounded * (sampling_period / dt))

    print(
        f"{Nt_original} time steps are required to traverse the simulation twice at a time step of {dt} seconds, at a total duration of {Nt_original * dt} seconds."
    )
    print(
        f"This amounts to {Nt_at_sampling_frequency} samples at {sampling_frequency} Hz, and {Nt_at_sampling_frequency_rounded} samples after rounding to the nearest power of two."
    )
    print(
        f"In order to achieve this, {Nt} time steps are required at the simulation time step."
    )

    desc = SimulationDescription(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        dx=spatial_resolution,
        dy=spatial_resolution,
        dz=spatial_resolution,
        Npml=Npml,
        dt=dt,
        output_length=Nt_at_sampling_frequency_rounded,
        Nt=Nt,
        air_properties=air_properties,
        obstacle_properties=obstacle_properties,
        sensor_locations=sensor_locations,
        impulse_location=(sensor_center_x, sensor_center_y, sensor_center_z),
    )

    return desc


def random_obstacle_single():
    r = random.randrange(1, 10)
    x = random.randrange(minimum_x_distance + r, Nx - r)
    y = random.randrange(r, Ny - r)
    z = random.randrange(r, Nz - r)
    return make_ball(Nx, Ny, Nz, x, y, z, r)


def random_obstacles():
    mask = random_obstacle_single()
    for _ in range(random.randint(0, 3)):
        mask |= random_obstacle_single()
    return mask


def main():
    desc = a_nice_description()

    with h5py.File("dataset_v1.h5", "w-") as h5file:
        init_dataset(h5file, desc)

    for i in range(100):
        si = str(i).zfill(4)
        print(f"##############################################################")
        print(f"#                                                            #")
        print(f"#                Creating Dataset Example {si}               #")
        print(f"#                                                            #")
        print(f"##############################################################")
        desc.set_obstacles(random_obstacles())
        results = run_kwave_simulation(desc)

        with h5py.File("dataset_v1.h5", "r+") as h5file:
            add_example_to_dataset(h5file, desc, results)


if __name__ == "__main__":
    main()
