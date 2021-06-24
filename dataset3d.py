from tof_utils import obstacle_map_to_sdf
from assert_eq import assert_eq
import h5py
import numpy as np
import torch

from h5ds import H5DS
from simulation_description import SimulationDescription
from device_dict import DeviceDict


class WaveDataset3d(torch.utils.data.Dataset):
    def __init__(self, description, path_to_h5file):
        super(WaveDataset3d, self).__init__()
        assert isinstance(description, SimulationDescription)
        assert isinstance(path_to_h5file, str)
        self.description = description
        # IMPORTANT: don't open the file in truncate mode or the dataset will be gone
        self.h5file = h5py.File(path_to_h5file, "a")
        assert self.h5file

        self.Nx = H5DS(name="Nx", dtype=np.uint32)
        self.Ny = H5DS(name="Ny", dtype=np.uint32)
        self.Nz = H5DS(name="Nz", dtype=np.uint32)
        self.dx = H5DS(name="dx", dtype=np.float32)
        self.dy = H5DS(name="dy", dtype=np.float32)
        self.dz = H5DS(name="dz", dtype=np.float32)

        self.air_speed_of_sound = H5DS(name="air_speed_of_sound", dtype=np.float32)
        self.signal_sampling_frequency = H5DS(
            name="signal_sampling_frequency", dtype=np.float32
        )
        self.signal_length = H5DS(name="signal_length", dtype=np.uint32)

        self.sensor_count = H5DS(name="sensor_count", dtype=np.uint32)
        self.sensor_indices = H5DS(
            name="sensor_locations",
            dtype=np.uint32,
            shape=(description.sensor_count, 3),
        )
        self.emitter_location = H5DS(
            name="emitter_location", dtype=np.uint32, shape=(3,)
        )

        self.sensor_recordings = H5DS(
            name="sensor_recordings",
            dtype=np.float32,
            shape=(description.sensor_count, description.output_length),
            extensible=True,
        )
        self.obstacles = H5DS(
            name="obstacles",
            dtype=np.bool8,
            shape=(
                description.Nx,
                description.Ny,
                description.Nz,
            ),
            extensible=True,
        )
        self.signed_distance_fields = H5DS(
            name="signed_distance_fields",
            dtype=np.float32,
            shape=(
                description.Nx,
                description.Ny,
                description.Nz,
            ),
            extensible=True,
        )

        if len(self.h5file.keys()) == 0:
            self._create_empty_dataset()

        self.validate()

    def _create_empty_dataset(self):
        assert self.h5file, "The file must be open"
        assert len(self.h5file.keys()) == 0, "The file must be empty"
        assert len(self.h5file.attrs.keys()) == 0, "The file must be empty"

        self.Nx.create(self.h5file, self.description.Nx)
        self.Ny.create(self.h5file, self.description.Ny)
        self.Nz.create(self.h5file, self.description.Nz)

        self.dx.create(self.h5file, self.description.dx)
        self.dy.create(self.h5file, self.description.dy)
        self.dz.create(self.h5file, self.description.dz)

        self.air_speed_of_sound.create(
            self.h5file, self.description.air_properties.speed_of_sound
        )
        self.signal_sampling_frequency.create(
            self.h5file, self.description.output_sampling_frequency
        )
        self.signal_length.create(self.h5file, self.description.output_length)

        self.sensor_count.create(self.h5file, self.description.sensor_count)
        self.sensor_indices.create(self.h5file, self.description.sensor_indices)
        self.emitter_location.create(self.h5file, self.description.emitter_indices)

        self.sensor_recordings.create(self.h5file)
        self.obstacles.create(self.h5file)
        self.signed_distance_fields.create(self.h5file)

    def validate(self):
        assert self.h5file, "The file must be open"

        assert_eq(self.Nx.read(self.h5file), self.description.Nx)
        assert_eq(self.Ny.read(self.h5file), self.description.Ny)
        assert_eq(self.Nz.read(self.h5file), self.description.Nz)

        assert_eq(self.dx.read(self.h5file), self.description.dx)
        assert_eq(self.dy.read(self.h5file), self.description.dy)
        assert_eq(self.dz.read(self.h5file), self.description.dz)

        assert_eq(
            self.air_speed_of_sound.read(self.h5file),
            self.description.air_properties.speed_of_sound,
        )
        assert_eq(
            self.signal_sampling_frequency.read(self.h5file),
            self.description.output_sampling_frequency,
        )

        assert_eq(self.signal_length.read(self.h5file), self.description.output_length)
        assert_eq(self.sensor_count.read(self.h5file), self.description.sensor_count)

        assert_eq(
            self.sensor_indices.read(self.h5file),
            self.description.sensor_indices,
        )
        assert_eq(
            self.emitter_location.read(self.h5file), self.description.emitter_indices
        )

        assert self.sensor_recordings.exists(self.h5file)
        assert self.obstacles.exists(self.h5file)
        assert self.signed_distance_fields.exists(self.h5file)

        assert_eq(
            self.sensor_recordings.count(self.h5file), self.obstacles.count(self.h5file)
        )

    def simulate_and_append_to_dataset(self, obstacles):
        assert self.h5file, "The file must be open"
        self.description.set_obstacles(obstacles)
        results = self.description.run()
        self.append_to_dataset(obstacles, results)

    def append_to_dataset(self, obstacles, recordings):
        assert self.h5file, "The file must be open"
        assert isinstance(obstacles, np.ndarray)
        assert_eq(obstacles.dtype, np.bool8)
        assert_eq(
            obstacles.shape,
            (
                self.description.Nx,
                self.description.Ny,
                self.description.Nz,
            ),
        )
        assert isinstance(recordings, np.ndarray)
        assert_eq(recordings.dtype, np.float32)
        assert_eq(
            recordings.shape,
            (self.description.sensor_count, self.description.output_length),
        )

        print("Computing signed distance field...")
        sdf = (
            obstacle_map_to_sdf(torch.tensor(obstacles).cuda(), self.description)
            .cpu()
            .numpy()
        )
        print("Computing signed distance field... done.")

        self.sensor_recordings.append(self.h5file, recordings)
        self.obstacles.append(self.h5file, obstacles)
        self.signed_distance_fields.append(self.h5file, sdf)
        self.validate()

    def __len__(self):
        assert self.h5file, "The file must be open"
        ret = self.sensor_recordings.count(self.h5file)
        return ret

    def __getitem__(self, idx):
        assert self.h5file, "The file must be open"
        sensor_recordings = self.sensor_recordings.read(self.h5file, idx)
        obstacles = self.obstacles.read(self.h5file, idx)
        sdf = self.signed_distance_fields.read(self.h5file, idx)
        sensor_recordings = torch.tensor(sensor_recordings)
        obstacles = torch.tensor(obstacles)
        sdf = torch.tensor(sdf)

        # Hmmmm
        sdf = torch.clamp(sdf, max=0.1)

        return DeviceDict(
            {"sensor_recordings": sensor_recordings, "obstacles": obstacles, "sdf": sdf}
        )
