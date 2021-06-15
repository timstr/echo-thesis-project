import torch
import torch.nn as nn
import torchvision

from simulation_description import SimulationDescription
from utils import assert_eq
from the_device import the_device
from time_of_flight_net import TimeOfFlightNet


def subset_recordings(recordings_batch, sensor_indices, description):
    assert isinstance(recordings_batch, torch.Tensor)
    assert isinstance(description, SimulationDescription)
    B = recordings_batch.shape[0]
    assert recordings_batch.shape == (
        B,
        description.sensor_count,
        description.output_length,
    )
    return recordings_batch[:, sensor_indices]


def simulation_extents_as_tensor(description, device):
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

    return locations_min, locations_max


def make_random_locations(batch_size, samples_per_example, device, description):
    locations_min, locations_max = simulation_extents_as_tensor(
        description, device=device
    )
    r = torch.rand(
        (batch_size, samples_per_example, 3), dtype=torch.float32, device=device
    )
    return (r * (locations_max - locations_min)) + locations_min


def sample_obstacle_map(obstacle_map_batch, locations_xyz_batch, description):
    assert isinstance(obstacle_map_batch, torch.Tensor)
    assert isinstance(locations_xyz_batch, torch.Tensor)
    assert isinstance(description, SimulationDescription)
    B = obstacle_map_batch.shape[0]
    assert_eq(
        obstacle_map_batch.shape,
        (
            B,
            description.Nx,
            description.Ny,
            description.Nz,
        ),
    )
    obstacle_map_batch = obstacle_map_batch.unsqueeze(1).to(torch.float32)

    # for grid_sample: batch, features, input depth, input height, input width
    assert_eq(
        obstacle_map_batch.shape,
        (
            B,
            1,
            description.Nx,
            description.Ny,
            description.Nz,
        ),
    )

    N = locations_xyz_batch.shape[1]
    assert_eq(locations_xyz_batch.shape, (B, N, 3))

    locations_min, locations_max = simulation_extents_as_tensor(
        description, device=locations_xyz_batch.device
    )

    t_0_1 = (locations_xyz_batch - locations_min) / (locations_max - locations_min)

    t_n1_1 = -1.0 + 2.0 * t_0_1
    assert_eq(t_n1_1.shape, (B, N, 3))

    # for grid_sample: batch, features, output depth, output height, output width, xyz coordinates
    t_grid = t_n1_1.reshape(B, N, 1, 1, 3)

    values = nn.functional.grid_sample(
        input=obstacle_map_batch,
        grid=t_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    assert_eq(values.shape, (B, 1, N, 1, 1))

    return values.reshape(B, N)


def plot_ground_truth(plt_axis, obstacle_map, description):
    assert isinstance(obstacle_map, torch.Tensor)
    assert isinstance(description, SimulationDescription)
    assert_eq(obstacle_map.shape, (description.Nx, description.Ny, description.Nz))

    num_slices = 10

    slices = []
    for i in range(num_slices):
        z = i * description.Nz // num_slices
        slices.append(obstacle_map[:, :, z].unsqueeze(0).float())

    img_grid = torchvision.utils.make_grid(
        tensor=slices, nrow=5, pad_value=0.5
    ).permute(2, 1, 0)
    plt_axis.imshow(img_grid)
    plt_axis.axis("off")


def plot_prediction(plt_axis, model, recordings, description):
    with torch.no_grad():
        assert isinstance(model, TimeOfFlightNet)
        assert isinstance(recordings, torch.Tensor)
        assert isinstance(description, SimulationDescription)
        C, L = recordings.shape
        recordings = recordings.unsqueeze(0)

        x_ls = torch.linspace(
            start=description.xmin, end=description.xmax, steps=description.Nx
        )
        y_ls = torch.linspace(
            start=description.ymin, end=description.ymax, steps=description.Ny
        )

        x_grid, y_grid = torch.meshgrid([x_ls, y_ls])
        z_grid = torch.zeros_like(y_grid)
        xyz = torch.stack([x_grid, y_grid, z_grid], dim=2).to(the_device)

        assert_eq(xyz.shape, (description.Nx, description.Ny, 3))

        xyz = xyz.reshape(1, (description.Nx * description.Ny), 3)

        num_slices = 10

        slices = []
        for i in range(num_slices):
            t = i / (num_slices - 1)
            z = description.zmin + t * (description.zmax - description.zmin)

            xyz[:, :2] = z

            prediction = model(recordings, xyz)

            assert prediction.shape == (1, (description.Nx * description.Ny))

            prediction = prediction.reshape(1, description.Nx, description.Ny)

            slices.append(prediction.cpu())

        img_grid = torchvision.utils.make_grid(
            tensor=slices, nrow=5, pad_value=0.5
        ).permute(2, 1, 0)
        plt_axis.imshow(img_grid)
        plt_axis.axis("off")
