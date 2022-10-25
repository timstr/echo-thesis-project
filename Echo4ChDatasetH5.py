import torch
import h5py
import numpy as np

from h5ds import H5DS
from device_dict import DeviceDict
from assert_eq import assert_eq

k_spectrograms = "spectrograms"
k_occupancy = "occupancy"
k_depthmap = "depthmap"


class Echo4ChDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_h5file, write=False):
        super(Echo4ChDataset, self).__init__()
        assert isinstance(path_to_h5file, str)
        assert isinstance(write, bool)

        # IMPORTANT: don't open the file in truncate mode or the dataset will be gone
        self.h5file = h5py.File(path_to_h5file, "a" if write else "r")
        assert self.h5file

        self.write = write

        self.spectrograms = H5DS(
            name="spectrograms", shape=(8, 256, 256), dtype=np.uint8, extensible=True
        )
        self.occupancy_maps = H5DS(
            name="occupancy", shape=(64, 64, 64), dtype=np.bool8, extensible=True
        )
        self.depthmaps = H5DS(
            name="depthmap", shape=(64, 64), dtype=np.uint8, extensible=True
        )

        if len(self.h5file.keys()) == 0 and write:
            self._create_empty_dataset()

        self.validate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self):
        self.h5file.close()

    def _create_empty_dataset(self):
        assert self.write, "The dataset must be opened with write=True"
        assert self.h5file, "The file must be open"
        assert len(self.h5file.keys()) == 0, "The file must be empty"
        assert len(self.h5file.attrs.keys()) == 0, "The file must be empty"

        self.spectrograms.create(self.h5file)
        self.occupancy_maps.create(self.h5file)
        self.depthmaps.create(self.h5file)

    def validate(self):
        assert self.h5file, "The file must be open"

        assert self.spectrograms.exists(self.h5file)
        assert self.occupancy_maps.exists(self.h5file)
        assert self.depthmaps.exists(self.h5file)

        N = self.spectrograms.count(self.h5file)

        assert_eq(self.occupancy_maps.count(self.h5file), N)

        assert_eq(self.depthmaps.count(self.h5file), N)

    def append_to_dataset(self, spectrograms, occupancy, depthmap):
        assert self.write, "The dataset must be opened with write=True"
        assert self.h5file, "The file must be open"
        assert isinstance(spectrograms, np.ndarray) or isinstance(
            spectrograms, torch.Tensor
        )
        assert spectrograms.dtype in [np.uint8, torch.uint8]
        assert_eq(spectrograms.shape, (8, 256, 256))
        assert isinstance(occupancy, np.ndarray) or isinstance(occupancy, torch.Tensor)
        assert occupancy.dtype in [np.bool8, torch.bool]
        assert_eq(occupancy.shape, (64, 64, 64))
        assert isinstance(depthmap, np.ndarray) or isinstance(depthmap, torch.Tensor)
        assert depthmap.dtype in [np.uint8, torch.uint8]
        assert_eq(depthmap.shape, (64, 64))

        self.spectrograms.append(self.h5file, spectrograms)
        self.occupancy_maps.append(self.h5file, occupancy)
        self.depthmaps.append(self.h5file, depthmap)
        self.validate()

    def __len__(self):
        assert self.h5file, "The file must be open"
        ret = self.spectrograms.count(self.h5file)
        return ret

    def __getitem__(self, idx):
        assert self.h5file, "The file must be open"
        spectrograms = torch.tensor(self.spectrograms.read(self.h5file, idx))
        occupancy = torch.tensor(self.occupancy_maps.read(self.h5file, idx))
        depthmap = torch.tensor(self.depthmaps.read(self.h5file, idx))

        return DeviceDict(
            {
                k_spectrograms: spectrograms,
                k_occupancy: occupancy,
                k_depthmap: depthmap,
            }
        )
