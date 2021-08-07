import torch
import torch.nn as nn
import numpy as np
import math

from utils import assert_eq, is_power_of_2
from reshape_layer import Reshape
from tof_utils import time_of_flight_crop

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
            data=torch.tensor(emitter_location, dtype=torch.float32),
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
            data=receiver_locations_tensor, requires_grad=False,
        )

        # Simple 2-layer fully-connected model
        # hidden_features = 256
        # self.model = nn.Sequential(
        #     nn.BatchNorm1d(num_features=num_receivers),
        #     Reshape(
        #         (num_receivers, crop_length_samples),
        #         (num_receivers * crop_length_samples,),
        #     ),
        #     nn.Linear(
        #         in_features=num_receivers * crop_length_samples,
        #         out_features=hidden_features,
        #     ),
        #     nn.ReLU(),
        #     nn.Linear(in_features=hidden_features, out_features=1),
        # )

        conv_features = 128
        final_length = crop_length_samples // 8
        final_features = 32

        self.model = nn.Sequential(
            nn.BatchNorm1d(num_features=num_receivers),
            nn.Conv1d(
                in_channels=num_receivers,
                out_channels=conv_features,
                kernel_size=15,
                stride=2,
                padding=7,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=conv_features),
            nn.Conv1d(
                in_channels=conv_features,
                out_channels=conv_features,
                kernel_size=15,
                stride=2,
                padding=7,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=conv_features),
            nn.Conv1d(
                in_channels=conv_features,
                out_channels=final_features,
                kernel_size=15,
                stride=2,
                padding=7,
            ),
            nn.ReLU(),
            Reshape((final_features, final_length), (final_features * final_length,)),
            nn.Linear(in_features=(final_features * final_length), out_features=1),
        )

    def forward(self, recordings, sample_locations):
        recordings_cropped = time_of_flight_crop(
            recordings=recordings,
            sample_locations=sample_locations,
            emitter_location=self.emitter_location,
            receiver_locations=self.receiver_locations,
            speed_of_sound=self.speed_of_sound,
            sampling_frequency=self.sampling_frequency,
            crop_length_samples=self.crop_length_samples,
        )

        recordings_cropped = sclog(recordings_cropped)

        B1, B2 = sample_locations.shape[:2]

        recordings_cropped_batches_combined = recordings_cropped.reshape(
            (B1 * B2), self.num_receivers, self.crop_length_samples
        )

        predictions = self.model(recordings_cropped_batches_combined)

        assert_eq(predictions.shape, ((B1 * B2), 1))
        predictions = predictions.reshape(B1, B2)

        return predictions
