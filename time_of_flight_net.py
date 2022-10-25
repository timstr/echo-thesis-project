import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

from utils import is_power_of_2
from assert_eq import assert_eq
from reshape_layer import Reshape
from signals_and_geometry import time_of_flight_crop


class TimeOfFlightNet(nn.Module):
    def __init__(
        self,
        speed_of_sound,
        sampling_frequency,
        recording_length_samples,
        crop_length_samples,
        emitter_location,
        receiver_locations,
        use_convolutions,
        hidden_features,
        kernel_size,
        use_fourier_transform,
        no_amplitude_compensation,
    ):
        super(TimeOfFlightNet, self).__init__()

        assert isinstance(speed_of_sound, float)
        assert isinstance(sampling_frequency, float)
        assert isinstance(recording_length_samples, int)
        assert is_power_of_2(recording_length_samples)
        assert isinstance(crop_length_samples, int)
        assert is_power_of_2(crop_length_samples)
        assert isinstance(emitter_location, np.ndarray)
        assert_eq(emitter_location.shape, (3,))
        assert_eq(emitter_location.dtype, np.float32)
        assert isinstance(receiver_locations, np.ndarray)
        assert receiver_locations.dtype == np.float32
        assert receiver_locations.shape[1:] == (3,)
        assert isinstance(use_convolutions, bool)
        assert isinstance(hidden_features, int)
        assert isinstance(kernel_size, int)
        assert isinstance(use_fourier_transform, bool)
        assert isinstance(no_amplitude_compensation, bool)

        self.speed_of_sound = speed_of_sound
        self.sampling_frequency = sampling_frequency
        self.recording_length_samples = recording_length_samples
        self.crop_length_samples = crop_length_samples
        self.emitter_location = nn.parameter.Parameter(
            data=torch.tensor(emitter_location, dtype=torch.float32),
            requires_grad=False,
        )

        num_receivers = receiver_locations.shape[0]
        self.num_receivers = num_receivers
        receiver_locations_tensor = torch.tensor(
            receiver_locations, dtype=torch.float32
        )
        assert receiver_locations_tensor.shape == (num_receivers, 3)
        self.receiver_locations = nn.parameter.Parameter(
            data=receiver_locations_tensor,
            requires_grad=False,
        )
        self.no_amplitude_compensation = no_amplitude_compensation

        self.use_fourier_transform = use_fourier_transform

        if self.use_fourier_transform:
            self.input_length = (crop_length_samples // 2) + 1
            self.channels_per_receiver = 2
        else:
            self.input_length = crop_length_samples
            self.channels_per_receiver = 1

        def activation_function():
            # return nn.ReLU()
            return nn.LeakyReLU(0.1)

        if use_convolutions:
            final_length = self.input_length // 8 + (
                1 if self.use_fourier_transform else 0
            )
            final_features = 1024 // final_length
            conv_padding = (kernel_size - 1) // 2

            self.model = nn.Sequential(
                nn.BatchNorm1d(num_features=num_receivers * self.channels_per_receiver),
                nn.Conv1d(
                    in_channels=num_receivers * self.channels_per_receiver,
                    out_channels=hidden_features,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=conv_padding,
                ),
                activation_function(),
                nn.BatchNorm1d(num_features=hidden_features),
                nn.Conv1d(
                    in_channels=hidden_features,
                    out_channels=hidden_features,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=conv_padding,
                ),
                activation_function(),
                nn.BatchNorm1d(num_features=hidden_features),
                nn.Conv1d(
                    in_channels=hidden_features,
                    out_channels=final_features,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=conv_padding,
                ),
                activation_function(),
                Reshape(
                    (final_features, final_length), (final_features * final_length,)
                ),
                nn.Linear(in_features=(final_features * final_length), out_features=1),
            )
        else:
            # Simple 3-layer fully-connected model
            self.model = nn.Sequential(
                nn.BatchNorm1d(num_features=num_receivers * self.channels_per_receiver),
                Reshape(
                    (num_receivers * self.channels_per_receiver, self.input_length),
                    (num_receivers * self.channels_per_receiver * self.input_length,),
                ),
                nn.Linear(
                    in_features=num_receivers
                    * self.channels_per_receiver
                    * self.input_length,
                    out_features=hidden_features,
                ),
                activation_function(),
                nn.Linear(
                    in_features=hidden_features,
                    out_features=hidden_features,
                ),
                activation_function(),
                nn.Linear(in_features=hidden_features, out_features=1),
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
            apply_amplitude_correction=(not self.no_amplitude_compensation),
        )

        B1, B2 = sample_locations.shape[:2]

        inputs = recordings_cropped.reshape(
            (B1 * B2), self.num_receivers, self.crop_length_samples
        )

        if self.use_fourier_transform:
            inputs = fft.rfft(
                inputs, n=self.crop_length_samples, dim=2, norm="backward"
            )
            assert_eq(
                inputs.shape,
                (B1 * B2, self.num_receivers, (self.crop_length_samples // 2) + 1),
            )
            inputs = torch.cat([torch.real(inputs), torch.imag(inputs)], dim=1)
            assert_eq(
                inputs.shape,
                (B1 * B2, 2 * self.num_receivers, (self.crop_length_samples // 2) + 1),
            )

        assert_eq(
            inputs.shape,
            (
                B1 * B2,
                self.num_receivers * self.channels_per_receiver,
                self.input_length,
            ),
        )

        predictions = self.model(inputs)

        assert_eq(predictions.shape, ((B1 * B2), 1))
        predictions = predictions.reshape(B1, B2)

        return predictions
