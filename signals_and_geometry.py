import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simulation_description import SimulationDescription

from assert_eq import assert_eq
from utils import progress_bar


# signed, clipped logarithm
def sclog(t):
    max_val = 1e0
    min_val = 1e-5
    signs = torch.sign(t)
    t = torch.abs(t)
    t = torch.clamp(t, min=min_val, max=max_val)
    t = torch.log(t)
    t = (t - math.log(min_val)) / (math.log(max_val) - math.log(min_val))
    t = t * signs
    return t


def make_fm_chirp(
    begin_frequency_Hz,
    end_frequency_Hz,
    sampling_frequency,
    chirp_length_samples,
    wave="sine",
    device="cpu",
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
    return torch.tensor(output, dtype=torch.float32, device=device)


def convolve_recordings(fm_chirp, sensor_recordings):
    with torch.no_grad():
        assert isinstance(fm_chirp, torch.Tensor)
        assert isinstance(sensor_recordings, torch.Tensor)
        (L_chirp,) = fm_chirp.shape
        batch_mode = sensor_recordings.ndim == 3
        if not batch_mode:
            sensor_recordings = sensor_recordings.unsqueeze(0)
        B, R, L_recording = sensor_recordings.shape
        assert L_chirp <= L_recording
        sensor_recordings_padded = torch.cat(
            [
                torch.zeros((B, R, L_chirp - 1), device=sensor_recordings.device),
                sensor_recordings,
            ],
            dim=2,
        )
        L_recording_padded = L_recording + L_chirp - 1
        assert_eq(sensor_recordings_padded.shape, (B, R, L_recording_padded))
        result = F.conv1d(
            input=sensor_recordings_padded.reshape(B * R, 1, L_recording_padded),
            weight=fm_chirp.flip(0).reshape(1, 1, L_chirp),
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        assert_eq(result.shape, (B * R, 1, L_recording))
        if batch_mode:
            return result.reshape(B, R, L_recording)
        return result.reshape(R, L_recording)


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


def sample_obstacle_map(obstacle_map_batch, locations_xyz_batch, description):
    assert isinstance(obstacle_map_batch, torch.Tensor)
    assert isinstance(locations_xyz_batch, torch.Tensor)
    assert isinstance(description, SimulationDescription)
    assert (obstacle_map_batch.ndim, locations_xyz_batch.ndim) in [(3, 2), (4, 3)]
    batch_mode = obstacle_map_batch.ndim == 4
    if not batch_mode:
        obstacle_map_batch = obstacle_map_batch.unsqueeze(0)
        locations_xyz_batch = locations_xyz_batch.unsqueeze(0)
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

    locations_min = torch.tensor(
        [description.xmin, description.ymin, description.zmin],
        dtype=torch.float32,
        device=locations_xyz_batch.device,
    ).reshape(1, 1, 3)

    locations_max = torch.tensor(
        [description.xmax, description.ymax, description.zmax],
        dtype=torch.float32,
        device=locations_xyz_batch.device,
    ).reshape(1, 1, 3)

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

    values = values.reshape(B, N)

    if not batch_mode:
        values = values.squeeze(0)

    return values


def time_of_flight_crop(
    recordings,
    sample_locations,
    emitter_location,
    receiver_locations,
    speed_of_sound,
    sampling_frequency,
    crop_length_samples,
    apply_amplitude_correction=True,
    center_time_of_arrival=True,
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

    center_time_of_arrival
        Whether to place the expected time of arrival at the center of the returned audio.
        Otherwise, it is placed at the beginning.
        bool, defaults to True
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

    crop_start_samples = total_samples

    if center_time_of_arrival:
        crop_start_samples = crop_start_samples - (crop_length_samples // 2)

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


def sdf_to_occupancy(sdf, threshold=0.0):
    assert isinstance(sdf, torch.Tensor)
    assert_eq(sdf.dtype, torch.float32)
    assert isinstance(threshold, float)
    return sdf <= threshold


def backfill_occupancy(occupancy):
    assert isinstance(occupancy, torch.Tensor)
    assert_eq(occupancy.dtype, torch.bool)
    assert occupancy.ndim in [3, 4]
    batch_mode = occupancy.ndim == 4
    if not batch_mode:
        occupancy = occupancy.unsqueeze(0)
    B, Nx, Ny, Nz = occupancy.shape
    mask = torch.zeros((B, Ny, Nz), dtype=torch.bool, device=occupancy.device)
    ret = torch.zeros_like(occupancy)
    for x in range(Nx):
        mask.logical_or_(occupancy[:, x])
        ret[:, x] = mask
    if not batch_mode:
        ret = ret.squeeze(0)
    return ret


def backfill_depthmap(depthmap, Nx):
    assert isinstance(depthmap, torch.Tensor)
    assert_eq(depthmap.dtype, torch.float32)
    assert isinstance(Nx, int)
    assert Nx > 0
    Ny, Nz = depthmap.shape
    ret = torch.zeros((Nx, Ny, Nz), dtype=torch.bool, device=depthmap.device)
    for x in range(Nx):
        # t = x / (Nx - 1)
        t = x / Nx
        ret[x] = depthmap <= t
    return ret
