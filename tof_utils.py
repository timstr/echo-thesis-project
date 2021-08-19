import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import math

from simulation_description import SimulationDescription
from utils import assert_eq, progress_bar
from current_simulation_description import minimum_x_units
from the_device import the_device


class SplitSize:
    def __init__(self, name):
        assert isinstance(name, str)
        self._num_splits = 1
        self._name = name

    def get(self):
        return self._num_splits

    def name(self):
        return self._name

    def double(self):
        self._num_splits *= 2


def split_till_it_fits(fn, split_size, *args, **kwargs):
    assert isinstance(split_size, SplitSize)
    split_size_was_increased = False
    max_size = 1024 * 1024
    recoverable_exceptions = [
        "out of memory",
        "not enough memory",
        "This error may appear if you passed in a non-contiguous input",
    ]
    while split_size.get() <= max_size:
        try:
            ret = fn(*args, **kwargs, num_splits=split_size.get())
            if split_size_was_increased:
                print(
                    f'The split size for "{split_size.name()}" was increased to {split_size.get()}'
                )
            return ret
        except RuntimeError as e:
            se = str(e)
            oom = any([excp in se for excp in recoverable_exceptions])
            if not oom:
                raise e
            torch.cuda.empty_cache()
            split_size.double()
            split_size_was_increased = True
    raise Exception(
        f'The split size for "{split_size.name()}" was increased too much and there\'s probably a bug'
    )


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


def evaluate_prediction(sdf_pred, sdf_gt, description, downsample_factor):
    assert isinstance(sdf_pred, torch.Tensor)
    assert isinstance(sdf_gt, torch.Tensor)
    assert isinstance(description, SimulationDescription)
    assert isinstance(downsample_factor, int)
    assert downsample_factor >= 1
    x_steps = (description.Nx - minimum_x_units) // downsample_factor
    y_steps = description.Ny // downsample_factor
    z_steps = description.Nz // downsample_factor
    assert_eq(sdf_pred.shape, (x_steps, y_steps, z_steps))
    assert_eq(sdf_gt.shape, (x_steps, y_steps, z_steps))

    mse_sdf = torch.mean((sdf_pred - sdf_gt) ** 2).item()

    occupancy_pred = (sdf_pred <= 0.0).bool()
    occupancy_gt = (sdf_gt <= 0.0).bool()

    gt_true = occupancy_gt
    gt_false = torch.logical_not(occupancy_gt)
    pred_true = occupancy_pred
    pred_false = torch.logical_not(occupancy_pred)

    def as_fraction(t):
        assert isinstance(t, torch.BoolTensor) or isinstance(t, torch.cuda.BoolTensor)
        f = torch.mean(t.float()).item()
        assert f >= 0.0 and f <= 1.0
        return f

    intersection = torch.logical_and(gt_true, pred_true)
    union = torch.logical_or(gt_true, pred_true)

    f_intersection = as_fraction(intersection)
    f_union = as_fraction(union)

    assert f_intersection >= 0.0
    assert f_intersection <= 1.0
    assert f_union >= 0.0
    assert f_union <= 1.0
    assert f_intersection <= f_union

    epsilon = 1e-6

    intersection_over_union = (f_intersection / f_union) if (f_union > epsilon) else 1.0

    assert intersection_over_union <= 1.0

    true_positives = torch.logical_and(gt_true, pred_true)
    true_negatives = torch.logical_and(gt_false, pred_false)
    false_positives = torch.logical_and(gt_false, pred_true)
    false_negatives = torch.logical_and(gt_true, pred_false)

    f_true_positives = as_fraction(true_positives)
    f_true_negatives = as_fraction(true_negatives)
    f_false_positives = as_fraction(false_positives)
    f_false_negatives = as_fraction(false_negatives)

    assert (
        abs(
            f_true_positives
            + f_true_negatives
            + f_false_positives
            + f_false_negatives
            - 1.0
        )
        < epsilon
    )

    selected = f_true_positives + f_false_positives
    relevant = f_true_positives + f_false_negatives

    precision = f_true_positives / selected if (selected > epsilon) else 0.0
    recall = f_true_positives / relevant if (relevant > epsilon) else 0.0

    f1score = (
        (2.0 * precision * recall / (precision + recall))
        if (abs(precision + recall) > epsilon)
        else 0.0
    )

    return {
        "mean_squared_error_sdf": mse_sdf,
        "intersection": f_intersection,
        "union": f_union,
        "intersection_over_union": intersection_over_union,
        "true_positives": f_true_positives,
        "true_negatives": f_true_negatives,
        "false_positives": f_false_positives,
        "false_negatives": f_false_negatives,
        "precision": precision,
        "recall": recall,
        "f1score": f1score,
    }


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


def _render_volumetric_slices(
    model,
    recordings,
    obstacle_map,
    description,
    num_splits,
    colour_function,
    locations=None,
):
    with torch.no_grad():
        assert model is None or isinstance(model, nn.Module)
        assert recordings is None or isinstance(recordings, torch.Tensor)
        assert obstacle_map is None or isinstance(obstacle_map, torch.Tensor)
        assert (model is None) != (obstacle_map is None)
        assert (model is None) == (recordings is None)
        assert isinstance(description, SimulationDescription)
        assert isinstance(num_splits, int)

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
            xyz = xyz.reshape((description.Nx * description.Ny), 3)

            if model is not None:
                prediction = split_network_prediction(
                    model=model,
                    locations=xyz,
                    recordings=recordings,
                    description=description,
                    num_splits=num_splits,
                )
            else:
                prediction = sample_obstacle_map(
                    obstacle_map.unsqueeze(0), xyz.unsqueeze(0), description
                ).squeeze(0)

            assert_eq(prediction.shape, (description.Nx * description.Ny,))
            prediction = prediction.reshape(description.Nx, description.Ny)

            prediction = colour_function(prediction)
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

            slices.append(prediction)

        img_grid = torchvision.utils.make_grid(tensor=slices, nrow=5, pad_value=0.5)
        return img_grid.permute(0, 2, 1)


def render_slices_ground_truth(
    obstacle_map, description, colour_function, locations=None
):
    return _render_volumetric_slices(
        model=None,
        recordings=None,
        obstacle_map=obstacle_map,
        description=description,
        num_splits=1,
        locations=locations,
        colour_function=colour_function,
    )


def split_network_prediction(model, locations, recordings, description, num_splits):
    assert isinstance(model, nn.Module)
    assert isinstance(locations, torch.Tensor)
    assert isinstance(recordings, torch.Tensor)
    assert isinstance(description, SimulationDescription)
    assert isinstance(num_splits, int)
    N, D = locations.shape
    locations = locations.reshape(1, N, D)
    assert_eq(D, 3)
    R, L = recordings.shape
    assert_eq(L, description.output_length)
    recordings = recordings.reshape(1, R, L)
    splits = []
    for i in range(num_splits):
        split_lo = N * i // num_splits
        split_hi = N * (i + 1) // num_splits
        xyz_split = locations[:, split_lo:split_hi]
        prediction_split = model(recordings=recordings, sample_locations=xyz_split)
        splits.append(prediction_split)
    prediction = torch.cat(splits, dim=1).squeeze(0)
    assert_eq(prediction.shape, (N,))
    return prediction


def render_slices_prediction(
    model, recordings, description, colour_function, num_splits
):
    return _render_volumetric_slices(
        model=model,
        recordings=recordings,
        obstacle_map=None,
        description=description,
        num_splits=num_splits,
        locations=None,
        colour_function=colour_function,
    )


def time_of_flight_crop(
    recordings,
    sample_locations,
    emitter_location,
    receiver_locations,
    speed_of_sound,
    sampling_frequency,
    crop_length_samples,
    apply_amplitude_correction=False,
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

    apply_amplitude_correction
        whether or not to increase the amplitude according the expected loss
        in signal strength due to the two travel paths and the single reflection.
        bool, defaults to False
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

    if apply_amplitude_correction:
        amplitude_compensation = (
            1000.0
            * torch.square(distance_emitter_to_target)
            * torch.square(distance_target_to_receivers)
        )
        assert_eq(amplitude_compensation.shape, (B1, B2, num_receivers))

        recordings_cropped_amplified = (
            recordings_cropped * amplitude_compensation.unsqueeze(-1)
        )

        return recordings_cropped_amplified

    return recordings_cropped


# def make_positive_distance_field(obstacle_map, description):
#     assert isinstance(obstacle_map, torch.Tensor)
#     assert_eq(obstacle_map.dtype, torch.bool)
#     assert isinstance(description, SimulationDescription)
#     assert_eq(obstacle_map.shape, (description.Nx, description.Ny, description.Nz))

#     kernel_radius = 2
#     kernel_size = 2 * kernel_radius + 1

#     distance_offsets = torch.empty(
#         (kernel_size, kernel_size, kernel_size), dtype=torch.float32
#     )
#     for i in range(kernel_size):
#         dx = (i - kernel_radius) * description.dx
#         for j in range(kernel_size):
#             dy = (j - kernel_radius) * description.dy
#             for k in range(kernel_size):
#                 dz = (k - kernel_radius) * description.dz
#                 d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
#                 distance_offsets[i, j, k] = d
#     distance_offsets = distance_offsets.reshape(
#         1, 1, 1, kernel_size, kernel_size, kernel_size
#     )
#     distance_offsets = distance_offsets.to(obstacle_map.device)

#     pad_sizes = 3 * (kernel_radius, kernel_radius)

#     # positive distance field
#     pdf = torch.full(
#         size=(description.Nx, description.Ny, description.Nz),
#         fill_value=np.inf,
#         dtype=torch.float32,
#         device=obstacle_map.device,
#     )
#     pdf[obstacle_map] = 0.0
#     num_iters = max(description.Nx, description.Ny, description.Nz) // kernel_radius
#     for current_iter in range(num_iters):
#         # pad with inf on each spatial axis
#         pdf_padded = F.pad(
#             pdf,
#             pad=pad_sizes,
#             mode="constant",
#             value=np.inf,
#         )
#         assert_eq(
#             pdf_padded.shape,
#             (
#                 description.Nx + 2 * kernel_radius,
#                 description.Ny + 2 * kernel_radius,
#                 description.Nz + 2 * kernel_radius,
#             ),
#         )
#         # unfold in each spatial dimension to create 3x3x3 kernels
#         pdf_unfolded = (
#             pdf_padded.unfold(dimension=0, size=kernel_size, step=1)
#             .unfold(dimension=1, size=kernel_size, step=1)
#             .unfold(dimension=2, size=kernel_size, step=1)
#         )
#         assert_eq(
#             pdf_unfolded.shape,
#             (
#                 description.Nx,
#                 description.Ny,
#                 description.Nz,
#                 kernel_size,
#                 kernel_size,
#                 kernel_size,
#             ),
#         )
#         # add distance offsets to each kernel value
#         pdf_offset = pdf_unfolded + distance_offsets
#         # take the minimum over each kernel
#         pdf_offset_flat = pdf_offset.reshape(
#             description.Nx, description.Ny, description.Nz, kernel_size ** 3
#         )
#         pdf_min, pdf_idx = torch.min(pdf_offset_flat, dim=3)
#         assert_eq(pdf_min.shape, (description.Nx, description.Ny, description.Nz))
#         pdf = pdf_min
#     return pdf


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

    print("Computing SDF from obstacle map")
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

        for i in range(kernel_size):
            x_lo = i
            x_hi = description.Nx + i
            for j in range(kernel_size):
                y_lo = j
                y_hi = description.Ny + j
                for k in range(kernel_size):
                    z_lo = k
                    z_hi = description.Nz + k

                    shifted = pdf_padded[x_lo:x_hi, y_lo:y_hi, z_lo:z_hi]

                    shifted_offset = shifted + distance_offsets[i, j, k].item()

                    pdf = torch.minimum(pdf, shifted_offset)

        assert_eq(
            pdf.shape,
            (
                description.Nx,
                description.Ny,
                description.Nz,
            ),
        )

        progress_bar(current_iter, num_iters)

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


def vector_cross(a, b):
    assert is_three_floats(a)
    assert is_three_floats(b)
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def vector_length(x):
    assert is_three_floats(x)
    return math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)


def vector_normalize(v, norm=1.0):
    assert is_three_floats(v)
    k = norm / vector_length(v)
    return [k * v[0], k * v[1], k * v[2]]


def _simulation_boundary_sdf(description, sample_locations, radius):
    assert isinstance(description, SimulationDescription)
    assert isinstance(sample_locations, torch.Tensor)
    D, N, M = sample_locations.shape
    assert D == 3
    assert isinstance(radius, float)

    locations_x = sample_locations[0]
    locations_y = sample_locations[1]
    locations_z = sample_locations[2]

    # --- x axes ---

    # compress x axis
    x_axes_locations_x_positive = torch.clamp(locations_x - description.xmax, min=0.0)
    x_axes_locations_x_negative = torch.clamp(locations_x - description.xmin, max=0.0)
    x_axes_locations_x = x_axes_locations_x_positive + x_axes_locations_x_negative

    # mirror and shift y axis
    x_axes_locations_y = torch.minimum(
        torch.abs(locations_y - description.ymin),
        torch.abs(locations_y - description.ymax),
    )

    # mirror and shift z axis
    x_axes_locations_z = torch.minimum(
        torch.abs(locations_z - description.zmin),
        torch.abs(locations_z - description.zmax),
    )

    # distance to point
    x_axes_locations = torch.stack(
        [x_axes_locations_x, x_axes_locations_y, x_axes_locations_z], dim=0
    )
    sdf_x_axes = torch.norm(x_axes_locations, dim=0) - radius

    # --- y axes ---

    # mirror and shift x axis
    y_axes_locations_x = torch.minimum(
        torch.abs(locations_x - description.xmin),
        torch.abs(locations_x - description.xmax),
    )

    # compress y axis
    y_axes_locations_y_positive = torch.clamp(locations_y - description.ymax, min=0.0)
    y_axes_locations_y_negative = torch.clamp(locations_y - description.ymin, max=0.0)
    y_axes_locations_y = y_axes_locations_y_positive + y_axes_locations_y_negative

    # mirror and shift z axis
    y_axes_locations_z = torch.minimum(
        torch.abs(locations_z - description.zmin),
        torch.abs(locations_z - description.zmax),
    )

    # distance to point
    y_axes_locations = torch.stack(
        [y_axes_locations_x, y_axes_locations_y, y_axes_locations_z], dim=0
    )
    sdf_y_axes = torch.norm(y_axes_locations, dim=0) - radius

    # --- x axes ---

    # mirror and shift z axis
    z_axes_locations_x = torch.minimum(
        torch.abs(locations_x - description.xmin),
        torch.abs(locations_x - description.xmax),
    )

    # mirror and shift y axis
    z_axes_locations_y = torch.minimum(
        torch.abs(locations_y - description.ymin),
        torch.abs(locations_y - description.ymax),
    )

    # compress x axis
    z_axes_locations_z_positive = torch.clamp(locations_z - description.zmax, min=0.0)
    z_axes_locations_z_negative = torch.clamp(locations_z - description.zmin, max=0.0)
    z_axes_locations_z = z_axes_locations_z_positive + z_axes_locations_z_negative

    # distance to point
    z_axes_locations = torch.stack(
        [z_axes_locations_x, z_axes_locations_y, z_axes_locations_z], dim=0
    )
    sdf_z_axes = torch.norm(z_axes_locations, dim=0) - radius

    return torch.minimum(
        torch.minimum(
            sdf_x_axes,
            sdf_y_axes,
        ),
        sdf_z_axes,
    )


def _raymarch_sdf_impl(
    camera_center_xyz,
    camera_up_xyz,
    camera_right_xyz,
    x_resolution,
    y_resolution,
    description,
    obstacle_sdf,
    model,
    recordings,
    num_splits,
):
    with torch.no_grad():
        assert is_three_floats(camera_center_xyz)
        assert is_three_floats(camera_up_xyz)
        assert is_three_floats(camera_right_xyz)
        assert isinstance(description, SimulationDescription)
        assert isinstance(num_splits, int)
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
        def make_tensor_3f(t, normalize=False):
            ret = torch.tensor([*t], dtype=torch.float32, device="cuda").reshape(
                3, 1, 1
            )
            if normalize:
                return ret / torch.norm(ret, dim=0, keepdim=True)
            return ret

        camera_center = make_tensor_3f(camera_center_xyz)
        camera_up = make_tensor_3f(camera_up_xyz)
        camera_right = make_tensor_3f(camera_right_xyz)
        camera_forward = make_tensor_3f(
            vector_cross(camera_up_xyz, camera_right_xyz), normalize=True
        )

        # create grid of view vectors using cross of two camera directions (and maybe offset from center for slight perspective)
        ls_x = torch.linspace(start=-1.0, end=1.0, steps=x_resolution, device="cuda")
        ls_y = torch.linspace(start=-1.0, end=1.0, steps=y_resolution, device="cuda")
        grid_x, grid_y = torch.meshgrid(ls_x, ls_y)
        offsets_x = grid_x.unsqueeze(0) * camera_right
        offsets_y = grid_y.unsqueeze(0) * camera_up
        locations = camera_center + offsets_x + offsets_y

        directions = camera_forward.repeat(1, x_resolution, y_resolution)

        # Add perspective distortion
        # directions = directions + 0.1 * (offsets_x + offsets_y)
        # directions /= torch.norm(directions, dim=0, keepdim=True)

        def _sample_obstacle_sdf(l):
            assert_eq(l.shape, (3, x_resolution, y_resolution))
            l_flat = l.reshape(1, 3, x_resolution * y_resolution).permute(0, 2, 1)
            assert l_flat.shape == (1, x_resolution * y_resolution, 3)
            if prediction:
                # num_splits = 256  # 128
                split_size = (x_resolution * y_resolution) // num_splits
                values_acc = []
                for i in range(num_splits):
                    idx_lo = i * split_size
                    idx_hi = (i + 1) * split_size
                    values_acc.append(
                        model(recordings.unsqueeze(0), l_flat[:, idx_lo:idx_hi])
                    )
                sdf_values = torch.cat(values_acc, dim=1)
            else:
                sdf_values = sample_obstacle_map(
                    obstacle_map_batch=obstacle_sdf.unsqueeze(0),
                    locations_xyz_batch=l_flat,
                    description=description,
                )
            assert_eq(sdf_values.shape, (1, x_resolution * y_resolution))
            x_in_bounds = (l[0] >= description.xmin).logical_and(
                l[0] <= description.xmax
            )
            y_in_bounds = (l[1] >= description.ymin).logical_and(
                l[1] <= description.ymax
            )
            z_in_bounds = (l[2] >= description.zmin).logical_and(
                l[2] <= description.zmax
            )
            in_bounds = x_in_bounds.logical_and(y_in_bounds).logical_and(z_in_bounds)
            out_of_bounds = in_bounds.logical_not()
            sdf_values = sdf_values.reshape(x_resolution, y_resolution)
            sdf_values[out_of_bounds] = 0.1
            return sdf_values

        # keep a boolean mask of rays that have not yet collided
        active = torch.ones(
            (x_resolution, y_resolution), dtype=torch.bool, device="cuda"
        )

        hit_axes = torch.zeros(
            (x_resolution, y_resolution), dtype=torch.bool, device="cuda"
        )

        num_iterations = 64
        for i in range(num_iterations):
            # get SDF values at each ray location
            sampled_sdf_obstacles = _sample_obstacle_sdf(locations)

            sampled_sdf_axes = _simulation_boundary_sdf(
                description, locations, radius=0.001
            )

            sampled_sdf = torch.minimum(sampled_sdf_obstacles, sampled_sdf_axes)

            # if SDF value is below threshold, make inactive
            threshold = 0.001
            active[sampled_sdf <= threshold] = 0
            hit_axes[sampled_sdf_axes <= threshold] = 1

            # advance all active rays by their direction vector times their SDF value
            locations[:, active] += (sampled_sdf * directions)[:, active]

            progress_bar(i, num_iterations)

        ret = torch.zeros(
            (3, x_resolution, y_resolution), dtype=torch.float32, device="cuda"
        )

        # fill non-collided pixels with background colour
        ret[:, active] = 1.0

        inactive = active.logical_not()

        # shade collide pixels with x,y,z partial derivatives of SDF at sampling locations
        h = 0.02
        dx = make_tensor_3f([0.5 * h, 0.0, 0.0])
        dy = make_tensor_3f([0.0, 0.5 * h, 0.0])
        dz = make_tensor_3f([0.0, 0.0, 0.5 * h])

        dsdfdx = (1.0 / h) * (
            _sample_obstacle_sdf(locations + dx) - _sample_obstacle_sdf(locations - dx)
        )
        dsdfdy = (1.0 / h) * (
            _sample_obstacle_sdf(locations + dy) - _sample_obstacle_sdf(locations - dy)
        )
        dsdfdz = (1.0 / h) * (
            _sample_obstacle_sdf(locations + dz) - _sample_obstacle_sdf(locations - dz)
        )
        assert_eq(dsdfdx.shape, (x_resolution, y_resolution))
        assert_eq(dsdfdy.shape, (x_resolution, y_resolution))
        assert_eq(dsdfdz.shape, (x_resolution, y_resolution))
        sdf_normal = torch.stack([dsdfdx, dsdfdy, dsdfdz], dim=0)
        sdf_normal /= torch.clamp(torch.norm(sdf_normal, dim=0, keepdim=True), min=1e-3)
        assert_eq(sdf_normal.shape, (3, x_resolution, y_resolution))
        light_dir = make_tensor_3f([-0.25, -1.0, 0.5], normalize=True)
        normal_dot_light = torch.sum(sdf_normal * light_dir, dim=0)
        assert_eq(normal_dot_light.shape, (x_resolution, y_resolution))
        shading = 0.2 + 0.6 * torch.clamp(normal_dot_light, min=0.0)

        ret[0][inactive] = shading[inactive]
        ret[1][inactive] = shading[inactive]
        ret[2][inactive] = shading[inactive]

        # colour axes
        ret[0][hit_axes] = 0.0
        ret[1][hit_axes] = 0.0
        ret[2][hit_axes] = 0.0

        return ret


def raymarch_sdf_ground_truth(
    camera_center_xyz,
    camera_up_xyz,
    camera_right_xyz,
    x_resolution,
    y_resolution,
    description,
    obstacle_sdf,
    num_splits,
):
    return _raymarch_sdf_impl(
        camera_center_xyz=camera_center_xyz,
        camera_up_xyz=camera_up_xyz,
        camera_right_xyz=camera_right_xyz,
        x_resolution=x_resolution,
        y_resolution=y_resolution,
        description=description,
        obstacle_sdf=obstacle_sdf,
        model=None,
        recordings=None,
        num_splits=num_splits,
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
    num_splits,
):
    return _raymarch_sdf_impl(
        camera_center_xyz=camera_center_xyz,
        camera_up_xyz=camera_up_xyz,
        camera_right_xyz=camera_right_xyz,
        x_resolution=x_resolution,
        y_resolution=y_resolution,
        description=description,
        obstacle_sdf=None,
        model=model,
        recordings=recordings,
        num_splits=num_splits,
    )


def make_fm_chirp(
    begin_frequency_Hz,
    end_frequency_Hz,
    sampling_frequency,
    chirp_length_samples,
    wave="sine",
):
    assert isinstance(begin_frequency_Hz, float)
    assert isinstance(end_frequency_Hz, float)
    assert isinstance(sampling_frequency, float)
    assert isinstance(chirp_length_samples, int)
    assert max(begin_frequency_Hz, end_frequency_Hz) <= (
        0.5 * sampling_frequency
    ), "Aliasing will occur"
    assert wave in ["sine", "square"]
    phase = 0.0
    output = np.zeros((chirp_length_samples,))
    for i in range(chirp_length_samples):
        t = i / (chirp_length_samples - 1)
        k = 8.0
        tt = min(abs(k * t), abs(k * (1.0 - t)), 1)
        a = 0.5 - 0.5 * math.cos(math.pi * tt)
        f = (1.0 - t) * begin_frequency_Hz + t * end_frequency_Hz
        phase += f / sampling_frequency
        phase -= math.floor(phase)
        output[i] = a * math.sin(phase * math.tau)
    if wave == "square":
        output = -1.0 + 2.0 * np.round(0.5 + 0.5 * output)
    return output


def convolve_recordings(fm_chirp, sensor_recordings, description):
    with torch.no_grad():
        assert isinstance(fm_chirp, torch.Tensor)
        assert isinstance(sensor_recordings, torch.Tensor)
        assert isinstance(description, SimulationDescription)
        (L_chirp,) = fm_chirp.shape
        batch_mode = sensor_recordings.ndim == 3
        if not batch_mode:
            sensor_recordings = sensor_recordings.unsqueeze(0)
        B, R, L_recording = sensor_recordings.shape
        assert_eq(L_recording, description.output_length)
        assert L_chirp <= L_recording
        chirp_padded = torch.cat(
            [torch.zeros((L_chirp - 1,), device=fm_chirp.device), fm_chirp], dim=0
        )
        L_chirp_padded = 2 * L_chirp - 1
        assert_eq(chirp_padded.shape, (L_chirp_padded,))
        result = F.conv1d(
            input=sensor_recordings.reshape(B * R, 1, L_recording),
            weight=chirp_padded.flip(dims=[0]).reshape(1, 1, L_chirp_padded),
            bias=None,
            stride=1,
            padding="same",
            dilation=1,
            groups=1,
        )
        assert_eq(result.shape, (B * R, 1, L_recording))
        if batch_mode:
            return result.reshape(B, R, L_recording)
        return result.reshape(R, L_recording)
