import math
import random

from simulation_description import AcousticMediumProperties, SimulationDescription
from kwave_util import make_ball, make_box

Npml = 10  # spatial count

Nx = 256 - 2 * Npml
Ny = (16 * 7) - 2 * Npml
Nz = (16 * 7) - 2 * Npml

minimum_x_units = Ny

spatial_resolution = 0.0075  # meters

sensor_count_x = 4
sensor_count_y = 4
sensor_count_z = 4


def make_simulation_description():

    c_air = 343.0
    c_wood = 4000.0
    # c_human = 1540.0
    # c_dense_air = c_air * 2.0

    rho_air = 1.225
    rho_wood = 500.0
    # rho_human = 1010.0
    # rho_dense_air = rho_air * 2.0

    sensor_center_x = minimum_x_units // 2
    sensor_center_y = Ny // 2
    sensor_center_z = Nz // 2

    # The sensors cover half the available distance
    sensor_extent_x = 0.5 * minimum_x_units
    sensor_extent_y = 0.5 * Ny
    sensor_extent_z = 0.5 * Nz

    sensor_spacing_x = sensor_extent_x / (sensor_count_x - 1)
    sensor_spacing_y = sensor_extent_y / (sensor_count_y - 1)
    sensor_spacing_z = sensor_extent_z / (sensor_count_z - 1)

    sensor_indices = []

    # record a small grid
    for i in range(sensor_count_x):
        for j in range(sensor_count_y):
            for k in range(sensor_count_z):
                x = round(
                    sensor_center_x - (sensor_extent_x / 2) + sensor_spacing_x * i
                )
                y = round(
                    sensor_center_y - (sensor_extent_y / 2) + sensor_spacing_y * j
                )
                z = round(
                    sensor_center_z - (sensor_extent_z / 2) + sensor_spacing_z * k
                )
                sensor_indices.append((x, y, z))

    air_properties = AcousticMediumProperties(
        speed_of_sound=c_air,  # meters per second
        density=rho_air,  # kilograms per cubic meter
    )
    obstacle_properties = AcousticMediumProperties(
        speed_of_sound=c_wood,  # meters per second
        density=rho_wood,  # kilograms per cubic meter
        # speed_of_sound=c_wood,  # meters per second
        # density=rho_wood,  # kilograms per cubic meter
    )

    dt = 2e-7  # seconds
    # dt = 1e-6  # seconds

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

    # # HACK
    # print("HACK: reduced timesteps for testing")
    # Nt = Nt // 10

    # print(
    #     f"{Nt_original} time steps are required to traverse the simulation twice at a time step of {dt} seconds, at a total duration of {Nt_original * dt} seconds."
    # )
    # print(
    #     f"This amounts to {Nt_at_sampling_frequency} samples at {sampling_frequency} Hz, and {Nt_at_sampling_frequency_rounded} samples after rounding to the nearest power of two."
    # )
    # print(
    #     f"In order to achieve this, {Nt} time steps are required at the simulation time step."
    # )

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
        sensor_indices=sensor_indices,
        emitter_indices=(sensor_center_x, sensor_center_y, sensor_center_z),
    )

    return desc


SHAPE_TYPE_SPHERE = "sphere"
SHAPE_TYPE_BOX = "box"


def make_random_obstacle_single(description):
    assert isinstance(description, SimulationDescription)
    min_radius = math.ceil(0.01 / spatial_resolution)
    max_radius = math.ceil(0.10 / spatial_resolution)
    shape_type = random.choice([SHAPE_TYPE_SPHERE, SHAPE_TYPE_BOX])
    if shape_type == SHAPE_TYPE_SPHERE:
        r = random.randrange(min_radius, max_radius)
        x = random.randrange(minimum_x_units + r, Nx - r)
        y = random.randrange(r, Ny - r)
        z = random.randrange(r, Nz - r)
        return make_ball(Nx, Ny, Nz, x, y, z, r)
    elif shape_type == SHAPE_TYPE_BOX:
        rx = random.randrange(min_radius, max_radius)
        ry = random.randrange(min_radius, max_radius)
        rz = random.randrange(min_radius, max_radius)
        cx = random.randrange(minimum_x_units + rx, Nx - rx)
        cy = random.randrange(ry, Ny - ry)
        cz = random.randrange(rz, Nz - rz)
        return make_box(Nx, Ny, Nz, cx, cy, cz, rx, ry, rz)
    else:
        raise Exception("What???")


def make_random_obstacles(description):
    assert isinstance(description, SimulationDescription)
    mask = make_random_obstacle_single(description)
    for _ in range(random.randint(0, 3)):
        mask |= make_random_obstacle_single(description)
    return mask
