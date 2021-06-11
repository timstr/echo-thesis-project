import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import h5py


def slice_along_dim(arr, dim, start=None, stop=None, step=None):
    assert isinstance(arr, np.ndarray)
    assert isinstance(dim, int) and dim >= 0 and dim < arr.ndim
    begin = (slice(None),) * dim
    middle = (slice(start, stop, step),)
    end = (slice(None),) * (arr.ndim - dim - 1)
    all_slices = begin + middle + end
    assert len(all_slices) == arr.ndim
    return arr[all_slices]


def stagger_along_dim(arr, dim):
    assert isinstance(arr, np.ndarray)
    assert isinstance(dim, int) and dim >= 0 and dim < arr.ndim
    arr_offset = np.concatenate(
        [
            slice_along_dim(arr, dim=dim, start=1),
            slice_along_dim(arr, dim=dim, start=-1),
        ],
        axis=dim,
    )
    assert arr_offset.shape == arr.shape
    return 0.5 * (arr + arr_offset)


def make_ball(Nx, Ny, Nz, centerx, centery, centerz, radius):
    x, y, z = np.meshgrid(
        np.linspace(start=0, stop=Nx, num=Nx, endpoint=False),
        np.linspace(start=0, stop=Ny, num=Ny, endpoint=False),
        np.linspace(start=0, stop=Nz, num=Nz, endpoint=False),
        indexing="ij",
    )
    ret = np.zeros(shape=(Nx, Ny, Nz), dtype=np.bool8)
    ret[(x - centerx) ** 2 + (y - centery) ** 2 + (z - centerz) ** 2 <= radius ** 2] = 1
    return ret


def make_box(Nx, Ny, Nz, centerx, centery, centerz, radiusx, radiusy, radiusz):
    assert isinstance(Nx, int)
    assert isinstance(Ny, int)
    assert isinstance(Nz, int)
    assert isinstance(centerx, int)
    assert isinstance(centery, int)
    assert isinstance(centerz, int)
    assert isinstance(radiusx, int)
    assert isinstance(radiusy, int)
    assert isinstance(radiusz, int)
    ret = np.zeros(shape=(Nx, Ny, Nz), dtype=np.bool8)
    ret[
        centerx - radiusx : centerx + radiusx,
        centery - radiusy : centery + radiusy,
        centerz - radiusz : centerz + radiusz,
    ] = 1
    return ret


def make_3d_blackman_window_via_outer_product(Nx, Ny, Nz):
    window_x = signal.windows.blackman(Nx, sym=(Nx % 2 == 1))
    window_y = signal.windows.blackman(Ny, sym=(Ny % 2 == 1))
    window_z = signal.windows.blackman(Nz, sym=(Nz % 2 == 1))
    window_3d = np.cbrt(
        np.abs(
            window_x[:, np.newaxis, np.newaxis]
            * window_y[np.newaxis, :, np.newaxis]
            * window_z[np.newaxis, np.newaxis, :]
        )
    )
    assert window_3d.shape == (Nx, Ny, Nz)
    return window_3d


def make_3d_blackman_window_via_rotation(Nx, Ny, Nz):
    x, y, z = np.meshgrid(
        np.linspace(start=-1.0, stop=1.0, num=Nx, endpoint=False),
        np.linspace(start=-1.0, stop=1.0, num=Ny, endpoint=False),
        np.linspace(start=-1.0, stop=1.0, num=Nz, endpoint=False),
        indexing="ij",
    )
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    a0 = 0.42
    a1 = 0.5
    a2 = 0.08
    n_over_N = np.clip(0.5 + 0.5 * rad, a_min=0.0, a_max=1.0)
    pi_n_over_N = np.pi * n_over_N
    window = a0 - a1 * np.cos(2 * pi_n_over_N) + a2 * np.cos(4 * pi_n_over_N)
    # return np.clip(window, a_min=0.0, a_max=1.0)
    return window


def smooth(p0):
    # print(f"Before smoothing:")
    # print(f"  min(p0) = {np.min(p0)}")
    # print(f"  max(p0) = {np.max(p0)}")
    old_max = np.max(p0)
    Nx, Ny, Nz = p0.shape
    # half_window_3d = make_3d_blackman_window_via_outer_product(Nx, Ny, Nz)
    half_window_3d = make_3d_blackman_window_via_rotation(Nx, Ny, Nz)
    assert half_window_3d.shape == (Nx, Ny, Nz)
    half_window_3d = fft.ifftshift(half_window_3d)
    p0_fd = fft.fftn(p0, norm="ortho")
    assert p0_fd.shape == (Nx, Ny, Nz)
    p0_fd_windowed = p0_fd * half_window_3d
    p0_smoothed = np.real(fft.ifftn(p0_fd_windowed, norm="ortho"))
    new_max = np.max(p0_smoothed)
    p0_smoothed = p0_smoothed * (old_max / new_max)
    # print(f"After smoothing:")
    # print(f"  min(p0) = {np.min(p0_smoothed)}")
    # print(f"  max(p0) = {np.max(p0_smoothed)}")
    assert p0_smoothed.shape == (Nx, Ny, Nz)
    return p0_smoothed


def make_scalar_for_kwave(x, dtype):
    return np.array([[[x]]], dtype=dtype)


utf8_type = h5py.string_dtype("utf-8", 128)


def encode_str(s):
    assert len(s) < 128
    return np.array(s.encode("utf-8"), dtype=utf8_type)


def get_dtype_string(dtype):
    if dtype == np.float32:
        return "float"
    elif dtype == np.uint64:
        return "long"
    else:
        raise Exception(f"unrecognized dtype: {dtype}")


def write_array(h5file, name, arr, dtype):
    assert isinstance(h5file, h5py.File)
    assert isinstance(name, str)
    assert isinstance(arr, np.ndarray)
    arr = arr.astype(dtype)
    ds = h5file.create_dataset(name, shape=arr.shape, dtype=dtype, data=arr)


def make_empty_extensible_dataset(h5file, name, shape, dtype):
    assert isinstance(h5file, h5py.File)
    assert isinstance(name, str)
    assert isinstance(shape, tuple)
    ds = h5file.create_dataset(
        name,
        shape=(0, *shape),
        maxshape=(None, *shape),
        dtype=dtype,
        compression="gzip",
        compression_opts=9,
    )


def append_to_dataset(h5file, name, arr, dtype):
    assert isinstance(h5file, h5py.File)
    assert isinstance(name, str)
    assert isinstance(arr, np.ndarray)
    ds = h5file[name]
    assert ds.dtype == dtype
    assert ds.shape[1:] == arr.shape
    N = ds.shape[0]
    ds.resize(size=(N + 1), axis=0)
    ds[-1] = arr


def write_scalar(h5file, name, value, dtype):
    assert isinstance(h5file, h5py.File)
    assert isinstance(name, str)
    assert isinstance(value, int) or isinstance(value, float)
    arr = np.array([value], dtype=dtype)
    ds = h5file.create_dataset(name, shape=arr.shape, dtype=dtype, data=arr)


def read_scalar(h5file, name):
    assert isinstance(h5file, h5py.File)
    assert isinstance(name, str)
    ds = h5file[name]
    assert ds.shape == (1,)
    return ds[0]


def write_array_for_kwave(h5file, name, arr, dtype):
    assert isinstance(h5file, h5py.File)
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 3
    # arr = arr.transpose(2, 1, 0)
    magic_threshold = 2 ** 17
    chunk_dims = (1,) + tuple(
        [s if s <= magic_threshold else s // 2 for s in arr.shape[1:]]
    )
    arr = arr.astype(dtype)
    ds = h5file.create_dataset(
        name, shape=arr.shape, dtype=dtype, data=arr, chunks=chunk_dims
    )
    ds.attrs["data_type"] = encode_str(get_dtype_string(dtype))
    ds.attrs["domain_type"] = encode_str("real")


def write_scalar_for_kwave(h5file, name, value, dtype):
    assert isinstance(h5file, h5py.File)
    assert isinstance(value, int) or isinstance(value, float)
    # h5file[name] = scalar(value, dtype=dtype)
    # h5file[name].attrs["data_type"] = encode_str(get_dtype_string(dtype))
    # h5file[name].attrs["domain_type"] = encode_str("real")
    arr = make_scalar_for_kwave(value, dtype=dtype)
    write_array_for_kwave(h5file, name, arr, dtype)
