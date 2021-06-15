import torch
import torch.nn as nn
import numpy as np

from utils import assert_eq, is_power_of_2


class TimeOfFlightNet(nn.Module):
    def __init__(
        self,
        speed_of_sound,
        sampling_frequency,
        recording_length_samples,
        crop_length_samples,
        emitter_location,
        receiver_locations,
    ):
        super(TimeOfFlightNet, self).__init__()

        assert isinstance(speed_of_sound, float)
        self.speed_of_sound = speed_of_sound

        assert isinstance(sampling_frequency, float)
        self.sampling_frequency = sampling_frequency

        assert isinstance(recording_length_samples, int)
        assert is_power_of_2(recording_length_samples)
        self.recording_length_samples = recording_length_samples

        assert isinstance(crop_length_samples, int)
        assert is_power_of_2(crop_length_samples)
        self.crop_length_samples = crop_length_samples

        assert isinstance(emitter_location, np.ndarray)
        assert_eq(emitter_location.shape, (3,))
        assert_eq(emitter_location.dtype, np.float32)
        self.emitter_location = nn.parameter.Parameter(
            data=torch.tensor(emitter_location, dtype=torch.float32).reshape(
                1, 1, 1, 3
            ),
            requires_grad=False,
        )

        assert isinstance(receiver_locations, np.ndarray)
        assert receiver_locations.dtype == np.float32
        assert receiver_locations.shape[1:] == (3,)
        num_receivers = receiver_locations.shape[0]
        self.num_receivers = num_receivers
        receiver_locations_tensor = torch.tensor(
            receiver_locations, dtype=torch.float32
        )
        assert receiver_locations_tensor.shape == (num_receivers, 3)
        self.receiver_locations = nn.parameter.Parameter(
            data=receiver_locations_tensor.reshape(1, 1, num_receivers, 3),
            requires_grad=False,
        )

        hidden_features = 32
        self.model = nn.Sequential(
            nn.Linear(
                in_features=num_receivers * crop_length_samples,
                out_features=hidden_features,
            ),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=1),
        )

    def forward(self, recordings, sample_locations):
        assert isinstance(recordings, torch.Tensor)
        assert isinstance(sample_locations, torch.Tensor)

        # First batch dimension: number of separate recordings
        B1 = recordings.shape[0]

        # Second batch dimension: number of sampling locations per recording
        B2 = sample_locations.shape[1]

        assert_eq(
            recordings.shape,
            (
                B1,
                self.num_receivers,
                self.recording_length_samples,
            ),
        )
        recordings = recordings.unsqueeze(1).repeat(1, B2, 1, 1)
        assert_eq(
            recordings.shape,
            (
                B1,
                B2,
                self.num_receivers,
                self.recording_length_samples,
            ),
        )

        assert_eq(sample_locations.shape, (B1, B2, 3))

        sample_locations = sample_locations.unsqueeze(2)
        assert_eq(sample_locations.shape, (B1, B2, 1, 3))

        distance_emitter_to_target = torch.norm(
            sample_locations - self.emitter_location, dim=3
        )
        distance_target_to_receivers = torch.norm(
            sample_locations - self.receiver_locations, dim=3
        )

        assert_eq(distance_emitter_to_target.shape, (B1, B2, 1))
        assert_eq(distance_target_to_receivers.shape, (B1, B2, self.num_receivers))

        total_distance = distance_emitter_to_target + distance_target_to_receivers
        assert_eq(total_distance.shape, (B1, B2, self.num_receivers))

        total_time = total_distance / self.speed_of_sound

        total_samples = torch.round(total_time * self.sampling_frequency)

        crop_start_samples = total_samples - self.crop_length_samples // 2
        assert_eq(crop_start_samples.shape, (B1, B2, self.num_receivers))

        crop_grid_base = -1.0 + 2.0 * (
            crop_start_samples / self.recording_length_samples
        )
        assert_eq(crop_grid_base.shape, (B1, B2, self.num_receivers))
        crop_grid_base = crop_grid_base.reshape(B1, B2, self.num_receivers, 1)

        crop_grid_offset = torch.linspace(
            start=0.0,
            # NOTE: total extent of grid is 2 units, hence multiplication by 0.5
            end=(0.5 * (self.crop_length_samples - 1) / self.recording_length_samples),
            steps=self.crop_length_samples,
            device=sample_locations.device,
        )
        assert_eq(crop_grid_offset.shape, (self.crop_length_samples,))
        crop_grid_offset = crop_grid_offset.reshape(1, 1, 1, self.crop_length_samples)

        crop_grid = crop_grid_base + crop_grid_offset
        assert_eq(
            crop_grid.shape, (B1, B2, self.num_receivers, self.crop_length_samples)
        )

        crop_grid = crop_grid.reshape(
            (B1 * B2 * self.num_receivers), self.crop_length_samples, 1
        )
        crop_grid = torch.stack([crop_grid, torch.zeros_like(crop_grid)], dim=-1)

        # for grid_sample: batch, height, width, 2
        assert_eq(
            crop_grid.shape,
            ((B1 * B2 * self.num_receivers), self.crop_length_samples, 1, 2),
        )

        # For grid_sample: batch, features, height, width
        recordings = recordings.reshape(
            (B1 * B2 * self.num_receivers), 1, self.recording_length_samples, 1
        )

        recordings_cropped = nn.functional.grid_sample(
            input=recordings, grid=crop_grid, mode="bilinear", align_corners=False
        )

        assert_eq(
            recordings_cropped.shape,
            ((B1 * B2 * self.num_receivers), 1, self.crop_length_samples, 1),
        )

        recordings_cropped = recordings_cropped.reshape(
            B1, B2, self.num_receivers, self.crop_length_samples
        )

        # Yikes

        predictions = self.model(
            recordings_cropped.reshape(
                (B1 * B2), (self.num_receivers * self.crop_length_samples)
            )
        )

        assert_eq(predictions.shape, ((B1 * B2), 1))
        predictions = predictions.reshape(B1, B2)

        return predictions
