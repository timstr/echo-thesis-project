import math
import random
import torch

from signals_and_geometry import sample_obstacle_map
from simulation_description import AcousticMediumProperties, SimulationDescription
from kwave_util import make_ball, make_box
from assert_eq import assert_eq

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


def make_receiver_indices(num_x, num_y, num_z):
    options = {1: [2], 2: [0, 3], 4: [0, 1, 2, 3]}
    assert num_x in options.keys()
    assert num_y in options.keys()
    assert num_z in options.keys()
    indices_x = options[num_x]
    indices_y = options[num_y]
    indices_z = options[num_z]
    flat_indices = []
    for ix in indices_x:
        for iy in indices_y:
            for iz in indices_z:
                flat_indices.append(ix * 16 + iy * 4 + iz)
    return flat_indices


def make_random_training_locations(
    obstacle_map_batch, samples_per_example, device, description
):
    with torch.no_grad():
        assert isinstance(obstacle_map_batch, torch.Tensor)
        assert isinstance(samples_per_example, int)
        assert isinstance(description, SimulationDescription)
        B = obstacle_map_batch.shape[0]
        locations_min = torch.tensor(
            [description.xmin, description.ymin, description.zmin],
            dtype=torch.float32,
            device=device,
        ).reshape(1, 1, 3)

        locations_max = torch.tensor(
            [description.xmax, description.ymax, description.zmax],
            dtype=torch.float32,
            device=device,
        ).reshape(1, 1, 3)

        def new_locations():
            r = torch.rand(
                (B, samples_per_example, 3), dtype=torch.float32, device=device
            )
            return locations_min + (r * (locations_max - locations_min))

        locations = new_locations()
        mask = torch.ones_like(locations, dtype=torch.bool)
        for _ in range(50):
            sdf = sample_obstacle_map(obstacle_map_batch, locations, description)
            mask[sdf < 0.099] = 0
            locations[mask] = new_locations()[mask]
        return locations


def all_grid_locations(device, description, downsample_factor):
    assert isinstance(description, SimulationDescription)
    assert isinstance(downsample_factor, int)
    assert downsample_factor >= 1
    xmin_location = (minimum_x_units - description.emitter_indices[0]) * description.dx
    x_steps = (description.Nx - minimum_x_units) // downsample_factor
    y_steps = description.Ny // downsample_factor
    z_steps = description.Nz // downsample_factor
    x_ls = torch.linspace(
        start=xmin_location,
        end=description.xmax,
        steps=x_steps,
        device=device,
    )
    y_ls = torch.linspace(
        start=description.ymin,
        end=description.ymax,
        steps=y_steps,
        device=device,
    )
    z_ls = torch.linspace(
        start=description.zmin,
        end=description.zmax,
        steps=z_steps,
        device=device,
    )
    gx, gy, gz = torch.meshgrid([x_ls, y_ls, z_ls])
    gx = gx.flatten()
    gy = gy.flatten()
    gz = gz.flatten()
    assert_eq(gx.shape, (x_steps * y_steps * z_steps,))
    assert_eq(gy.shape, (x_steps * y_steps * z_steps,))
    assert_eq(gz.shape, (x_steps * y_steps * z_steps,))
    all_locations = torch.stack([gx, gy, gz], dim=1)
    assert_eq(all_locations.shape, (x_steps * y_steps * z_steps, 3))
    return all_locations
