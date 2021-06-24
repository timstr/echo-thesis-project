import h5py
import numpy as np

from assert_eq import assert_eq


class H5DS:
    def __init__(self, name, dtype, shape=(1,), extensible=False):
        assert isinstance(name, str)
        assert isinstance(shape, tuple)
        assert all([isinstance(s, int) for s in shape])
        assert isinstance(extensible, bool)
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.extensible = extensible

    def create(self, h5file, value=None):
        assert isinstance(h5file, h5py.File)
        assert h5file
        assert not self.exists(h5file)
        if self.extensible:
            if value is None:
                shape = (0,) + self.shape
            else:
                if self.shape == (1,):
                    value = np.array([value], dtype=self.dtype)
                value = value[np.newaxis, :]
                shape = (1,) + self.shape
            maxshape = (None,) + self.shape
            chunks = (1,) + self.shape
        else:
            shape = self.shape
            maxshape = self.shape
            chunks = self.shape
            if self.shape == (1,):
                value = np.array([value], dtype=self.dtype)
            assert isinstance(value, np.ndarray)
            assert_eq(value.dtype, self.dtype)
            assert_eq(value.shape, self.shape)
        ds = h5file.create_dataset(
            name=self.name,
            shape=shape,
            maxshape=maxshape,
            chunks=chunks,
            dtype=self.dtype,
            # compression="gzip",
            # compression_opts=9,
        )
        ds[...] = value

    def read(self, h5file, index=None):
        assert isinstance(h5file, h5py.File)
        assert h5file
        assert self.exists(h5file)
        ds = h5file[self.name]
        if self.extensible:
            assert isinstance(index, int)
            assert index < ds.shape[0]
            if self.shape == (1,):
                return ds[index, 0]
            return np.array(ds[index])
        else:
            assert index is None
            if self.shape == (1,):
                return ds[0]
            return np.array(ds)

    def write(self, h5file, value, index=None):
        assert isinstance(h5file, h5py.File)
        assert h5file
        if self.shape == (1,):
            assert isinstance(value, int) or isinstance(value, float)
            value = np.array([value], dtype=self.dtype)
        assert isinstance(value, np.ndarray)
        assert_eq(value.dtype, self.dtype)
        assert_eq(value.shape, self.shape)
        assert self.exists(h5file)
        ds = h5file[self.name]
        if self.extensible:
            assert isinstance(index, int)
            assert index < ds.shape[0]
            ds[index] = value
        else:
            assert index is None
            ds[...] = value

    def append(self, h5file, value):
        assert isinstance(h5file, h5py.File)
        assert h5file
        if self.shape == (1,):
            assert isinstance(value, int) or isinstance(value, float)
            value = np.array([value], dtype=self.dtype)
        assert isinstance(value, np.ndarray)
        assert_eq(value.dtype, self.dtype)
        assert_eq(value.shape, self.shape)
        assert self.extensible
        assert self.exists(h5file)
        ds = h5file[self.name]
        N = ds.shape[0]
        ds.resize(size=(N + 1), axis=0)
        ds[-1] = value

    def count(self, h5file):
        assert isinstance(h5file, h5py.File)
        assert h5file
        assert self.extensible
        ds = h5file[self.name]
        return ds.shape[0]

    def exists(self, h5file):
        assert isinstance(h5file, h5py.File)
        assert h5file
        if self.name not in h5file.keys():
            return False
        ds = h5file[self.name]
        if ds.dtype != self.dtype:
            raise Exception(
                f"Incorrect dtype found in HDF5 dataset. Expected '{self.name}' to have type {self.dtype} but found {ds.dtype} instead."
            )
        if self.extensible:
            N = ds.shape[0]
            if ds.shape[1:] != self.shape:
                raise Exception(
                    f"Incorrect shape found in HDF5 dataset. Expected '{self.name}' to have extensible shape N*{'*'.join(self.shape)} but found {'*'.join(ds.shape)} instead."
                )
        else:
            if ds.shape != self.shape:
                raise Exception(
                    f"Incorrect shape found in HDF5 dataset. Expected '{self.name}' to have inextensible shape {'*'.join(self.shape)} but found {'*'.join(ds.shape)} instead."
                )
        return True
