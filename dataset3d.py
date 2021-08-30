from utils import progress_bar
from assert_eq import assert_eq
import h5py
import numpy as np
import torch
import hashlib

from h5ds import H5DS
from simulation_description import SimulationDescription
from device_dict import DeviceDict


class WaveDataset3d(torch.utils.data.Dataset):
    def __init__(self, description, path_to_h5file, write=False):
        super(WaveDataset3d, self).__init__()
        assert isinstance(description, SimulationDescription)
        assert isinstance(path_to_h5file, str)
        assert isinstance(write, bool)
        self.description = description
        # IMPORTANT: don't open the file in truncate mode or the dataset will be gone
        self.h5file = h5py.File(path_to_h5file, "a" if write else "r")
        self.write = write
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

        self.bytes_per_hash = 256 // 8
        self.obstacle_hashes = H5DS(
            name="obstacle_hashes",
            dtype=np.uint8,
            shape=(self.bytes_per_hash,),
            extensible=True,
        )

        self._obstacle_hashes_cache = []
        self._obstacle_hashes_cache_stale = True

        if len(self.h5file.keys()) == 0 and write:
            self._create_empty_dataset()

        self.validate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type == SystemExit:
            print("Dataset was closed due to SystemExit")
        return False

    def close(self):
        self.h5file.close()
        self._obstacle_hashes_cache = []
        self._obstacle_hashes_cache_stale = True

    def _create_empty_dataset(self):
        assert self.write, "The dataset must be opened with write=True"
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

        self.obstacle_hashes.create(self.h5file)

        self._obstacle_hashes_cache = []
        self._obstacle_hashes_cache_stale = False

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

        N = self.sensor_recordings.count(self.h5file)

        assert_eq(self.obstacles.count(self.h5file), N)

        assert_eq(self.signed_distance_fields.count(self.h5file), N)

        # HACK
        # TODO: assert that obstacle_hashes exists and always assert count
        if self.obstacle_hashes.exists(self.h5file):
            assert_eq(self.obstacle_hashes.count(self.h5file), N)
        else:
            print("WARNING: obstacle hashes not found in dataset")

    def append_to_dataset(self, obstacles, recordings, sdf, skip_duplicates=False):
        assert self.write, "The dataset must be opened with write=True"
        assert self.h5file, "The file must be open"
        assert isinstance(obstacles, np.ndarray) or isinstance(obstacles, torch.Tensor)
        assert obstacles.dtype in [np.bool8, torch.bool]
        assert_eq(
            obstacles.shape,
            (
                self.description.Nx,
                self.description.Ny,
                self.description.Nz,
            ),
        )
        assert isinstance(recordings, np.ndarray) or isinstance(
            recordings, torch.Tensor
        )
        assert recordings.dtype in [np.float32, torch.float32]
        assert_eq(
            recordings.shape,
            (self.description.sensor_count, self.description.output_length),
        )

        assert isinstance(sdf, np.ndarray) or isinstance(sdf, torch.Tensor)
        assert sdf.dtype in [np.float32, torch.float32]
        assert sdf.shape == (
            self.description.Nx,
            self.description.Ny,
            self.description.Nz,
        )

        hash_result = self._hash_obstacles(obstacles)

        self._update_obstacle_hash_cache()

        N = len(self._obstacle_hashes_cache)
        for i in range(N):
            other_hash = self._obstacle_hashes_cache[i]
            if np.all(hash_result == other_hash):
                if skip_duplicates:
                    return False
                raise Exception(
                    "Attempted to add a set of obstacles that were already present in the dataset"
                )

        self.obstacle_hashes.append(self.h5file, hash_result)

        assert not self._obstacle_hashes_cache_stale
        self._obstacle_hashes_cache.append(hash_result)

        self.sensor_recordings.append(self.h5file, recordings)
        self.obstacles.append(self.h5file, obstacles)
        self.signed_distance_fields.append(self.h5file, sdf)
        self.validate()

        return True

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

    def contains(self, obstacles):
        hash_result = self._hash_obstacles(obstacles)
        self._update_obstacle_hash_cache()
        N = len(self._obstacle_hashes_cache)
        for i in range(N):
            if np.all(hash_result == self._obstacle_hashes_cache[i]):
                return True
        return False

    def contains_any_duplicates(self):
        assert self.h5file, "The file must be open"

        self._update_obstacle_hash_cache()

        N = len(self._obstacle_hashes_cache)

        for i in range(N):
            for j in range(i + 1, N):
                if np.all(
                    self._obstacle_hashes_cache[i] == self._obstacle_hashes_cache[j]
                ):
                    return True
        return False

    def _hash_obstacles(self, obstacles):
        assert isinstance(obstacles, np.ndarray) or isinstance(obstacles, torch.Tensor)
        assert obstacles.dtype in [np.bool8, torch.bool]
        assert_eq(
            obstacles.shape,
            (
                self.description.Nx,
                self.description.Ny,
                self.description.Nz,
            ),
        )

        obstacles_packed = np.packbits(obstacles.flatten())
        assert isinstance(obstacles_packed, np.ndarray)
        assert_eq(obstacles_packed.dtype, np.uint8)

        hash_fn = hashlib.sha256()
        hash_fn.update(obstacles_packed.tobytes())
        hash_result = np.frombuffer(
            hash_fn.digest(), dtype=np.uint8, count=self.bytes_per_hash
        )

        return hash_result

    def _update_obstacle_hash_cache(self):
        if not self._obstacle_hashes_cache_stale:
            return
        N = self.obstacle_hashes.count(self.h5file)
        self._obstacle_hashes_cache = []
        print("Refreshing obstacle map cache...")
        for i in range(N):
            self._obstacle_hashes_cache.append(
                self.obstacle_hashes.read(self.h5file, i)
            )
            progress_bar(i, N)
        print("Refreshing obstacle map cache... Done")
        self._obstacle_hashes_cache_stale = False
