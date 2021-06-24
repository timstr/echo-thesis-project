import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import math

from simulation_description import SimulationDescription
from utils import assert_eq, progress_bar
from the_device import the_device


def subset_recordings(recordings_batch, sensor_indices, description):
    assert isinstance(recordings_batch, torch.Tensor)
    assert isinstance(description, SimulationDescription)
    B = recordings_batch.shape[0]
    assert_eq(
        recordings_batch.shape,
        (
            B,
            description.sensor_count,
            description.output_length,
        ),
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


def make_random_training_locations(
    obstacle_map_batch, samples_per_example, device, description
):
    with torch.no_grad():
        assert isinstance(obstacle_map_batch, torch.Tensor)
        assert isinstance(samples_per_example, int)
        assert isinstance(description, SimulationDescription)
        B = obstacle_map_batch.shape[0]
        locations_min, locations_max = simulation_extents_as_tensor(
            description, device=device
        )

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

    # HACK: excluding half because the emitter overpowers everything otherwise
    # locations_min[0] = 0.5 * (locations_max[0] - locations_min[0])
    # r = torch.rand(
    #     (batch_size, samples_per_example, 3), dtype=torch.float32, device=device
    # )
    # return (r * (locations_max - locations_min)) + locations_min


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

    # NOTE: grid_sample uses z,y,x indexing here
    t_n1_1 = t_n1_1.flip(dims=(2,))

    # for grid_sample: batch, features, output depth, output height, output width, xyz coordinates
    t_grid = t_n1_1.reshape(B, N, 1, 1, 3)

    values = nn.functional.grid_sample(
        input=obstacle_map_batch,
        grid=t_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )

    assert_eq(values.shape, (B, 1, N, 1, 1))

    return values.reshape(B, N)


def plot_ground_truth(plt_axis, obstacle_map, description, locations=None):
    with torch.no_grad():
        assert isinstance(description, SimulationDescription)

        x_ls = torch.linspace(
            start=description.xmin,
            end=description.xmax,
            steps=description.Nx,
            # device=the_device,
        )
        y_ls = torch.linspace(
            start=description.ymin,
            end=description.ymax,
            steps=description.Ny,
            # device=the_device,
        )

        x_grid, y_grid = torch.meshgrid([x_ls, y_ls])

        num_slices = 10

        slices = []
        for i in range(num_slices):
            t = i / (num_slices - 1)
            z = description.zmin + t * (description.zmax - description.zmin)

            z_grid = z * torch.ones_like(x_grid)
            xyz = torch.stack([x_grid, y_grid, z_grid], dim=2)  # .to(the_device)
            assert_eq(xyz.shape, (description.Nx, description.Ny, 3))
            xyz = xyz.reshape(1, (description.Nx * description.Ny), 3)

            # num_splits = 8
            # assert (description.Nx * description.Ny) % num_splits == 0
            # split_size = (description.Nx * description.Ny) // num_splits
            # splits = []
            # for i in range(num_splits):
            #     split_lo = i * split_size
            #     split_hi = (i + 1) * split_size
            #     xyz_split = xyz[:, split_lo:split_hi]
            #     prediction_split = sample_obstacle_map(
            #         obstacle_map.unsqueeze(0), xyz_split, description
            #     )
            #     splits.append(prediction_split)
            # prediction = torch.cat(splits, dim=1)
            prediction = sample_obstacle_map(
                obstacle_map.unsqueeze(0), xyz, description
            ).squeeze(0)

            assert_eq(prediction.shape, (description.Nx * description.Ny,))
            prediction = prediction.reshape(description.Nx, description.Ny)

            prediction = colourize_sdf(prediction)
            assert_eq(prediction.shape, (3, description.Nx, description.Ny))

            prediction = prediction.cpu()

            if locations is not None:
                for lx, ly, lz in locations:
                    if abs(z - lz) > (
                        (description.zmax - description.zmin) / num_slices
                    ):
                        continue
                    px = round(
                        (
                            (lx - description.xmin)
                            / (description.xmax - description.xmin)
                            * (description.Nx - 1)
                        ).item()
                    )
                    py = round(
                        (
                            (ly - description.ymin)
                            / (description.ymax - description.ymin)
                            * (description.Ny - 1)
                        ).item()
                    )
                    prediction[0, px, py] = 0.0
                    prediction[1, px, py] = 0.0
                    prediction[2, px, py] = 0.0

            # prediction = torch.clamp(prediction, min=0.0, max=1.0)
            # assert_eq(prediction.shape, (1, (description.Nx * description.Ny)))
            # prediction = prediction.reshape(1, description.Nx, description.Ny)

            slices.append(prediction)

        img_grid = torchvision.utils.make_grid(
            tensor=slices, nrow=5, pad_value=0.5
        ).permute(2, 1, 0)
        plt_axis.imshow(img_grid)
        plt_axis.axis("off")


def plot_prediction(plt_axis, model, recordings, description):
    with torch.no_grad():
        assert isinstance(model, nn.Module)
        assert isinstance(recordings, torch.Tensor)
        assert isinstance(description, SimulationDescription)
        C, L = recordings.shape
        recordings = recordings.unsqueeze(0)

        x_ls = torch.linspace(
            start=description.xmin,
            end=description.xmax,
            steps=description.Nx,
            device=the_device,
        )
        y_ls = torch.linspace(
            start=description.ymin,
            end=description.ymax,
            steps=description.Ny,
            device=the_device,
        )

        x_grid, y_grid = torch.meshgrid([x_ls, y_ls])

        num_slices = 10

        slices = []
        for i in range(num_slices):
            t = i / (num_slices - 1)
            z = description.zmin + t * (description.zmax - description.zmin)

            z_grid = z * torch.ones_like(x_grid)
            xyz = torch.stack([x_grid, y_grid, z_grid], dim=2).to(the_device)
            assert_eq(xyz.shape, (description.Nx, description.Ny, 3))
            xyz = xyz.reshape(1, (description.Nx * description.Ny), 3)

            num_splits = 8
            assert (description.Nx * description.Ny) % num_splits == 0
            split_size = (description.Nx * description.Ny) // num_splits
            splits = []
            for i in range(num_splits):
                split_lo = i * split_size
                split_hi = (i + 1) * split_size
                xyz_split = xyz[:, split_lo:split_hi]
                prediction_split = model(recordings, xyz_split)
                splits.append(prediction_split)
            prediction = torch.cat(splits, dim=1).squeeze(0)
            # prediction = model(recordings, xyz)

            assert_eq(prediction.shape, (description.Nx * description.Ny,))
            prediction = prediction.reshape(description.Nx, description.Ny)

            prediction = colourize_sdf(prediction)
            assert_eq(prediction.shape, (3, description.Nx, description.Ny))

            # prediction = torch.clamp(prediction, min=0.0, max=1.0)
            # assert_eq(prediction.shape, (1, (description.Nx * description.Ny)))
            # prediction = prediction.reshape(1, description.Nx, description.Ny)

            slices.append(prediction.cpu())

        img_grid = torchvision.utils.make_grid(
            tensor=slices, nrow=5, pad_value=0.5
        ).permute(2, 1, 0)
        plt_axis.imshow(img_grid)
        plt_axis.axis("off")


def time_of_flight_crop(
    recordings,
    sample_locations,
    emitter_location,
    receiver_locations,
    speed_of_sound,
    sampling_frequency,
    crop_length_samples,
):
    """
    recordings:
        Audio recordings at each receiver location, assumed to start
        exactly when emitter first produces sound and to have a sampling
        frequency given by `sampling_frequency`
        size (N x num_receivers x recording_length_samples) tensor of float32,
        where N is the batch dimension

    sample_locations:
        Coordinates in xyz space of locations to sample, in meters.
        size (N x M x 3) tensor of float32, where N is the ordinary batch
        dimension corresponding to that of `recordings`, and M is the number
        of sample locations per recording. This allows the separating the
        number of recordings per batch and the number of samples per recording.

    emitter_location:
        Coordinates in xyz space of the emitter, in meters.
        If in doubt, make this the origin and position sampling locations
        and receivers relative to the emitter.
        size (3) tensor of float32

    receiver_locations:
        Coordinates in xyz space of the receivers, in meters.
        size (num_receivers x 3) tensor of float32

    speed_of_sound:
        The speed of sound through air, in meters per second
        float

    sampling_frequency:
        The rate at which the receivers record, in Hertz
        float

    crop_length_samples
        The length in samples of the cropped audio
        int
    """

    assert isinstance(recordings, torch.Tensor)
    assert isinstance(sample_locations, torch.Tensor)
    assert isinstance(emitter_location, torch.Tensor)
    assert isinstance(receiver_locations, torch.Tensor)

    assert_eq(recordings.device, sample_locations.device)
    assert_eq(recordings.device, emitter_location.device)
    assert_eq(recordings.device, receiver_locations.device)
    device = recordings.device

    assert isinstance(speed_of_sound, float)
    assert isinstance(sampling_frequency, float)
    assert isinstance(crop_length_samples, int)

    # Validate recordings
    # First batch dimension: number of separate recordings
    B1, num_receivers, recording_length_samples = recordings.shape

    # Validate sample locations
    # Second batch dimension: number of sampling locations per recording
    B2 = sample_locations.shape[1]
    recordings = recordings.unsqueeze(1).repeat(1, B2, 1, 1)
    assert_eq(
        recordings.shape,
        (
            B1,
            B2,
            num_receivers,
            recording_length_samples,
        ),
    )

    # Validate emitter location
    assert_eq(emitter_location.shape, (3,))
    assert_eq(emitter_location.dtype, torch.float32)
    emitter_location = emitter_location.reshape(1, 1, 1, 3)

    # Validate receiver locations
    assert_eq(receiver_locations.dtype, torch.float32)
    assert_eq(
        receiver_locations.shape,
        (
            num_receivers,
            3,
        ),
    )
    receiver_locations = receiver_locations.reshape(1, 1, num_receivers, 3)

    assert_eq(sample_locations.shape, (B1, B2, 3))

    sample_locations = sample_locations.unsqueeze(2)
    assert_eq(sample_locations.shape, (B1, B2, 1, 3))

    distance_emitter_to_target = torch.norm(sample_locations - emitter_location, dim=3)
    distance_target_to_receivers = torch.norm(
        sample_locations - receiver_locations, dim=3
    )

    assert_eq(distance_emitter_to_target.shape, (B1, B2, 1))
    assert_eq(distance_target_to_receivers.shape, (B1, B2, num_receivers))

    total_distance = distance_emitter_to_target + distance_target_to_receivers
    assert_eq(total_distance.shape, (B1, B2, num_receivers))

    total_time = total_distance / speed_of_sound

    total_samples = torch.round(total_time * sampling_frequency)

    crop_start_samples = total_samples - (crop_length_samples // 2)
    assert_eq(crop_start_samples.shape, (B1, B2, num_receivers))

    crop_grid_per_receiver_offset = crop_start_samples.reshape(B1, B2, num_receivers, 1)

    crop_grid_per_sample_offset = torch.linspace(
        start=0.0,
        end=(crop_length_samples - 1),
        steps=crop_length_samples,
        device=device,
    )

    assert_eq(crop_grid_per_sample_offset.shape, (crop_length_samples,))
    crop_grid_per_sample_offset = crop_grid_per_sample_offset.reshape(
        1, 1, 1, crop_length_samples
    )

    crop_grid_samples = crop_grid_per_receiver_offset + crop_grid_per_sample_offset
    assert_eq(crop_grid_samples.shape, (B1, B2, num_receivers, crop_length_samples))

    crop_grid = -1.0 + 2.0 * (crop_grid_samples / recording_length_samples)

    crop_grid = crop_grid.reshape((B1 * B2 * num_receivers), crop_length_samples, 1)
    crop_grid = torch.stack([torch.zeros_like(crop_grid), crop_grid], dim=-1)

    # for grid_sample: batch, height, width, 2
    assert_eq(
        crop_grid.shape,
        ((B1 * B2 * num_receivers), crop_length_samples, 1, 2),
    )

    # For grid_sample: batch, features, height, width
    recordings = recordings.reshape(
        (B1 * B2 * num_receivers), 1, recording_length_samples, 1
    )

    recordings_cropped = nn.functional.grid_sample(
        input=recordings, grid=crop_grid, mode="bilinear", align_corners=False
    )

    assert_eq(
        recordings_cropped.shape,
        ((B1 * B2 * num_receivers), 1, crop_length_samples, 1),
    )

    recordings_cropped = recordings_cropped.reshape(
        B1, B2, num_receivers, crop_length_samples
    )

    return recordings_cropped


def make_positive_distance_field(obstacle_map, description):
    assert isinstance(obstacle_map, torch.Tensor)
    assert_eq(obstacle_map.dtype, torch.bool)
    assert isinstance(description, SimulationDescription)
    assert_eq(obstacle_map.shape, (description.Nx, description.Ny, description.Nz))

    kernel_radius = 2
    kernel_size = 2 * kernel_radius + 1

    distance_offsets = torch.empty(
        (kernel_size, kernel_size, kernel_size), dtype=torch.float32
    )
    for i in range(kernel_size):
        dx = (i - kernel_radius) * description.dx
        for j in range(kernel_size):
            dy = (j - kernel_radius) * description.dy
            for k in range(kernel_size):
                dz = (k - kernel_radius) * description.dz
                d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                distance_offsets[i, j, k] = d
    distance_offsets = distance_offsets.reshape(
        1, 1, 1, kernel_size, kernel_size, kernel_size
    )
    distance_offsets = distance_offsets.to(obstacle_map.device)

    pad_sizes = 3 * (kernel_radius, kernel_radius)

    # positive distance field
    pdf = torch.full(
        size=(description.Nx, description.Ny, description.Nz),
        fill_value=np.inf,
        dtype=torch.float32,
        device=obstacle_map.device,
    )
    pdf[obstacle_map] = 0.0
    num_iters = max(description.Nx, description.Ny, description.Nz) // kernel_radius
    for current_iter in range(num_iters):
        # pad with inf on each spatial axis
        pdf_padded = F.pad(
            pdf,
            pad=pad_sizes,
            mode="constant",
            value=np.inf,
        )
        assert_eq(
            pdf_padded.shape,
            (
                description.Nx + 2 * kernel_radius,
                description.Ny + 2 * kernel_radius,
                description.Nz + 2 * kernel_radius,
            ),
        )
        # unfold in each spatial dimension to create 3x3x3 kernels
        pdf_unfolded = (
            pdf_padded.unfold(dimension=0, size=kernel_size, step=1)
            .unfold(dimension=1, size=kernel_size, step=1)
            .unfold(dimension=2, size=kernel_size, step=1)
        )
        assert_eq(
            pdf_unfolded.shape,
            (
                description.Nx,
                description.Ny,
                description.Nz,
                kernel_size,
                kernel_size,
                kernel_size,
            ),
        )
        # add distance offsets to each kernel value
        pdf_offset = pdf_unfolded + distance_offsets
        # take the minimum over each kernel
        pdf_offset_flat = pdf_offset.reshape(
            description.Nx, description.Ny, description.Nz, kernel_size ** 3
        )
        pdf_min, pdf_idx = torch.min(pdf_offset_flat, dim=3)
        assert_eq(pdf_min.shape, (description.Nx, description.Ny, description.Nz))
        pdf = pdf_min
    return pdf


def obstacle_map_to_sdf(obstacle_map, description):
    with torch.no_grad():
        assert isinstance(obstacle_map, torch.Tensor)
        assert isinstance(description, SimulationDescription)
        assert_eq(obstacle_map.shape, (description.Nx, description.Ny, description.Nz))
        assert_eq(obstacle_map.dtype, torch.bool)
        sdf = make_positive_distance_field(obstacle_map, description)
        not_obstacle_map = obstacle_map.logical_not()
        negative_sdf = -make_positive_distance_field(not_obstacle_map, description)
        sdf[obstacle_map] = negative_sdf[obstacle_map]
        # assert torch.all(torch.isfinite(sdf))
        return sdf


def smoothstep(edge0, edge1, x):
    assert edge0 < edge1
    t = torch.clamp((x - edge0) / (edge1 - edge0), min=0.0, max=1.0)
    return t * t * (3.0 - 2.0 * t)


def blue_orange_sdf_colours(img):
    H, W = img.shape
    img = img.unsqueeze(0)

    def colour(r, g, b):
        return torch.tensor([r, g, b], dtype=torch.float, device=img.device).reshape(
            3, 1, 1
        )

    blue = colour(0.22, 0.33, 0.66)
    orange = colour(0.93, 0.48, 0.10)
    paler_blue = colour(0.50, 0.58, 0.82)
    paler_orange = colour(0.93, 0.87, 0.28)
    white = colour(1.0, 1.0, 1.0)

    sign = torch.sign(img)

    base_colour = blue * (0.5 - 0.5 * sign) + orange * (0.5 + 0.5 * sign)
    paler_colour = paler_blue * (0.5 - 0.5 * sign) + paler_orange * (0.5 + 0.5 * sign)
    mix = torch.exp(-4.0 * torch.abs(img))

    out = base_colour + mix * (paler_colour - base_colour)

    out *= 1.0 - 0.2 * torch.cos(60.0 * img) ** 4

    out = torch.lerp(out, white, 1.0 - smoothstep(0.0, 0.02, torch.abs(img)))

    return out


def colourize_sdf(img):
    return blue_orange_sdf_colours(img)


def is_three_floats(x):
    return len(x) == 3 and all([isinstance(xi, float) for xi in x])


def __raymarch_sdf_impl(
    camera_center_xyz,
    camera_up_xyz,
    camera_right_xyz,
    x_resolution,
    y_resolution,
    description,
    obstacle_sdf,
    model,
    recordings,
):
    with torch.no_grad():
        assert is_three_floats(camera_center_xyz)
        assert is_three_floats(camera_up_xyz)
        assert is_three_floats(camera_right_xyz)
        assert isinstance(description, SimulationDescription)
        if obstacle_sdf is not None:
            assert isinstance(obstacle_sdf, torch.Tensor)
            assert obstacle_sdf.shape == (
                description.Nx,
                description.Ny,
                description.Nz,
            )
            assert model is None
            assert recordings is None
            prediction = False
        else:
            assert isinstance(model, nn.Module)
            assert isinstance(recordings, torch.Tensor)
            assert len(recordings.shape) == 2
            assert recordings.shape[1] == description.output_length
            prediction = True
        # create grid of sampling points using meshgrid between two camera directions
        def make_tensor_3f(t):
            return torch.tensor([*t], dtype=torch.float32, device="cuda").reshape(
                3, 1, 1
            )

        camera_center = make_tensor_3f(camera_center_xyz)
        camera_up = make_tensor_3f(camera_up_xyz)
        camera_right = make_tensor_3f(camera_right_xyz)
        camera_forward = make_tensor_3f(
            [
                camera_up_xyz[1] * camera_right_xyz[2]
                - camera_up_xyz[2] * camera_right_xyz[1],
                camera_up_xyz[2] * camera_right_xyz[0]
                - camera_up_xyz[0] * camera_right_xyz[2],
                camera_up_xyz[0] * camera_right_xyz[1]
                - camera_up_xyz[1] * camera_right_xyz[0],
            ]
        )

        # create grid of view vectors using cross of two camera directions (and maybe offset from center for slight perspective)
        ls_x = torch.linspace(start=-1.0, end=1.0, steps=x_resolution, device="cuda")
        ls_y = torch.linspace(start=-1.0, end=1.0, steps=y_resolution, device="cuda")
        grid_x, grid_y = torch.meshgrid(ls_x, ls_y)
        offsets_x = grid_x.unsqueeze(0) * camera_right
        offsets_y = grid_y.unsqueeze(0) * camera_up
        locations = camera_center + offsets_x + offsets_y

        directions = camera_forward.repeat(1, x_resolution, y_resolution)
        # directions = directions + 0.05 * (offsets_x + offsets_y)
        # directions /= torch.norm(directions, dim=0, keepdim=True)

        def sample_sdf(l):
            D, W, H = l.shape
            assert_eq(D, 3)
            l = l.reshape(1, 3, W * H).permute(0, 2, 1)
            if prediction:
                num_splits = 16
                split_size = (W * H) // num_splits
                values_acc = []
                for i in range(num_splits):
                    idx_lo = i * split_size
                    idx_hi = (i + 1) * split_size
                    values_acc.append(
                        model(recordings.unsqueeze(0), l[:, idx_lo:idx_hi])
                    )
                sdf_values = torch.cat(values_acc, dim=1)
            else:
                sdf_values = sample_obstacle_map(
                    obstacle_map_batch=obstacle_sdf.unsqueeze(0),
                    locations_xyz_batch=l,
                    description=description,
                )
            assert_eq(sdf_values.shape, (1, W * H))
            return sdf_values.reshape(W, H)

        # keep a boolean mask of rays that have not yet collided
        active = torch.ones(
            (x_resolution, y_resolution), dtype=torch.bool, device="cuda"
        )

        num_iterations = 500
        for i in range(num_iterations):
            # get SDF values at each ray location
            sdf = sample_sdf(locations)

            # if SDF value is below threshold, make inactive
            active[sdf <= 0.01] = 0

            # advance all active rays by their direction vector times their SDF value
            locations[:, active] += (sdf * directions)[:, active]

            progress_bar(i, num_iterations)

        ret = torch.zeros(
            (3, x_resolution, y_resolution), dtype=torch.float32, device="cuda"
        )

        # fill non-collided pixels with background colour
        ret[:, active] = 1.0

        inactive = active.logical_not()

        # shade collide pixels with x,y,z partial derivatives of SDF at sampling locations
        h = 0.0001
        dx = make_tensor_3f([0.5 * h, 0.0, 0.0])
        dy = make_tensor_3f([0.0, 0.5 * h, 0.0])
        dz = make_tensor_3f([0.0, 0.0, 0.5 * h])

        dsdfdx = (1.0 / h) * (sample_sdf(locations + dx) - sample_sdf(locations - dx))
        dsdfdy = (1.0 / h) * (sample_sdf(locations + dy) - sample_sdf(locations - dy))
        dsdfdz = (1.0 / h) * (sample_sdf(locations + dz) - sample_sdf(locations - dz))
        assert_eq(dsdfdx.shape, (x_resolution, y_resolution))
        assert_eq(dsdfdy.shape, (x_resolution, y_resolution))
        assert_eq(dsdfdz.shape, (x_resolution, y_resolution))
        sdf_normal = torch.stack([dsdfdx, dsdfdy, dsdfdz], dim=0)
        sdf_normal /= torch.clamp(torch.norm(sdf_normal, dim=0, keepdim=True), min=1e-3)
        assert_eq(sdf_normal.shape, (3, x_resolution, y_resolution))
        light_dir = make_tensor_3f([0.0, -1.0, 0.0])
        normal_dot_light = torch.sum(sdf_normal * light_dir, dim=0)
        assert_eq(normal_dot_light.shape, (x_resolution, y_resolution))
        shading = 0.2 + 0.6 * torch.clamp(normal_dot_light, min=0.0)

        ret[0][inactive] = shading[inactive]
        ret[1][inactive] = shading[inactive]
        ret[2][inactive] = shading[inactive]

        return ret


def raymarch_sdf_ground_truth(
    camera_center_xyz,
    camera_up_xyz,
    camera_right_xyz,
    x_resolution,
    y_resolution,
    description,
    obstacle_sdf,
):
    return __raymarch_sdf_impl(
        camera_center_xyz=camera_center_xyz,
        camera_up_xyz=camera_up_xyz,
        camera_right_xyz=camera_right_xyz,
        x_resolution=x_resolution,
        y_resolution=y_resolution,
        description=description,
        obstacle_sdf=obstacle_sdf,
        model=None,
        recordings=None,
    )


def raymarch_sdf_prediction(
    camera_center_xyz,
    camera_up_xyz,
    camera_right_xyz,
    x_resolution,
    y_resolution,
    description,
    model,
    recordings,
):
    return __raymarch_sdf_impl(
        camera_center_xyz=camera_center_xyz,
        camera_up_xyz=camera_up_xyz,
        camera_right_xyz=camera_right_xyz,
        x_resolution=x_resolution,
        y_resolution=y_resolution,
        description=description,
        obstacle_sdf=None,
        model=model,
        recordings=recordings,
    )
