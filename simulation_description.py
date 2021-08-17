from assert_eq import assert_eq
import datetime
import os
import subprocess
from kwave_util import (
    encode_str,
    make_ball,
    smooth,
    stagger_along_dim,
    write_array_for_kwave,
    write_scalar_for_kwave,
)
import numpy as np
import scipy.interpolate as interpolate
import scipy.signal as signal

import h5py


class AcousticMediumProperties:
    def __init__(self, speed_of_sound, density):
        assert isinstance(speed_of_sound, float)
        assert isinstance(density, float)
        self.speed_of_sound = speed_of_sound
        self.density = density


class SimulationDescription:
    def __init__(
        self,
        Nx,
        Ny,
        Nz,
        dx,
        dy,
        dz,
        Npml,
        Nt,
        dt,
        output_length,
        air_properties,
        obstacle_properties,
        sensor_indices,
        emitter_indices,
    ):
        assert isinstance(Nx, int)
        assert isinstance(Ny, int)
        assert isinstance(Nz, int)
        assert isinstance(dx, float)
        assert isinstance(dy, float)
        assert isinstance(dz, float)
        assert isinstance(Npml, int)
        assert isinstance(Nt, int)
        assert isinstance(dt, float)
        assert isinstance(output_length, int)
        assert output_length > 1 and output_length <= Nt
        assert isinstance(air_properties, AcousticMediumProperties)
        assert isinstance(obstacle_properties, AcousticMediumProperties)
        assert isinstance(sensor_indices, list)
        assert all(
            [
                isinstance(i, int)
                and isinstance(j, int)
                and isinstance(k, int)
                and (i >= 0 and i < Nx)
                and (j >= 0 and j < Ny)
                and (k >= 0 and k < Nz)
                for i, j, k in sensor_indices
            ]
        )
        assert isinstance(emitter_indices, tuple)
        assert len(emitter_indices) == 3
        assert isinstance(emitter_indices[0], int)
        assert isinstance(emitter_indices[1], int)
        assert isinstance(emitter_indices[2], int)

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.Npml = Npml

        self.Nx_total = Nx + 2 * Npml
        self.Ny_total = Ny + 2 * Npml
        self.Nz_total = Nz + 2 * Npml

        self.Nt = Nt  # samples
        self.dt = dt  # seconds
        self.simulation_sampling_frequency = 1.0 / dt  # Herz
        self.simulation_duration = Nt * dt  # seconds
        self.output_length = output_length  # samples
        self.output_sampling_frequency = (
            output_length / self.simulation_duration
        )  # Herz

        self.air_properties = air_properties
        self.obstacle_properties = obstacle_properties
        self.sensor_count = len(sensor_indices)
        self.sensor_indices = np.array(
            [[x, y, z] for x, y, z in sensor_indices], dtype=np.uint32
        )
        self.sensor_indices_flat = np.array(
            [self.make_simulation_index_flat(x, y, z) for x, y, z in sensor_indices]
        )[np.newaxis, np.newaxis, :]
        assert self.sensor_indices_flat.shape == (1, 1, len(sensor_indices))
        self.emitter_indices = np.array(emitter_indices, dtype=np.uint32)

        self.xmin = -emitter_indices[0] * self.dx
        self.ymin = -emitter_indices[1] * self.dy
        self.zmin = -emitter_indices[2] * self.dz
        self.xmax = (self.Nx - emitter_indices[0]) * self.dx
        self.ymax = (self.Ny - emitter_indices[1]) * self.dy
        self.zmax = (self.Nz - emitter_indices[2]) * self.dz

        self.emitter_location = np.zeros((3,), dtype=np.float32)

        self.sensor_locations = np.array(
            [
                [
                    (x - emitter_indices[0]) * dx,
                    (y - emitter_indices[1]) * dy,
                    (z - emitter_indices[2]) * dz,
                ]
                for x, y, z in sensor_indices
            ],
            dtype=np.float32,
        )

        assert_eq(self.sensor_locations.shape, (self.sensor_count, 3))
        assert_eq(self.sensor_locations.dtype, np.float32)

        self.total_shape = (self.Nx_total, self.Ny_total, self.Nz_total)
        self.slice_inner = (
            slice(Npml, Npml + Nx),
            slice(Npml, Npml + Ny),
            slice(Npml, Npml + Nz),
        )

        self.c_ref = max(
            self.air_properties.speed_of_sound, self.obstacle_properties.speed_of_sound
        )

        p0_raw = np.zeros(self.total_shape, dtype=np.float32)
        p0_raw[self.slice_inner] = make_ball(
            Nx,
            Ny,
            Nz,
            emitter_indices[0],
            emitter_indices[1],
            emitter_indices[2],
            radius=2,
        )

        self.p0 = smooth(p0_raw)
        self.p0 = p0_raw

        self.has_obstacles = False

    def set_obstacles(self, obstacle_mask):
        assert isinstance(obstacle_mask, np.ndarray)
        assert obstacle_mask.shape == (self.Nx, self.Ny, self.Nz)

        self.obstacle_mask = obstacle_mask
        self.obstacle_mask_with_pml = np.zeros(self.total_shape, dtype=np.bool8)
        self.obstacle_mask_with_pml[self.slice_inner] = obstacle_mask

        self.c0 = self.air_properties.speed_of_sound * np.ones(
            self.total_shape, dtype=np.float32
        )
        self.c0[self.obstacle_mask_with_pml] = self.obstacle_properties.speed_of_sound

        self.rho0 = self.air_properties.density * np.ones(
            self.total_shape, dtype=np.float32
        )
        self.rho0[self.obstacle_mask_with_pml] = self.obstacle_properties.density

        self.rho0_sgx = stagger_along_dim(self.rho0, dim=2)
        self.rho0_sgy = stagger_along_dim(self.rho0, dim=1)
        self.rho0_sgz = stagger_along_dim(self.rho0, dim=0)

        self.has_obstacles = True

    def make_simulation_index_flat(self, x, y, z):
        return (
            1
            + (z + self.Npml)
            + self.Ny_total * (y + self.Npml)
            + self.Nz_total * self.Ny_total * (x + self.Npml)
        )

    def _write_kwave_input_file(self, hdf5_input_file_path):
        assert isinstance(hdf5_input_file_path, str)
        assert (
            self.has_obstacles
        ), "You need to add obstacles before creating a simulation"
        with h5py.File(
            hdf5_input_file_path, mode="w", libver=("earliest", "v108")
        ) as f:
            write_array_for_kwave(f, "c0", self.c0, dtype=np.float32)
            write_array_for_kwave(f, "rho0", self.rho0, dtype=np.float32)

            # density along staggered grids
            write_array_for_kwave(f, "rho0_sgx", self.rho0_sgx, dtype=np.float32)
            write_array_for_kwave(f, "rho0_sgy", self.rho0_sgy, dtype=np.float32)
            write_array_for_kwave(f, "rho0_sgz", self.rho0_sgz, dtype=np.float32)

            # initial pressure distribution
            write_array_for_kwave(f, "p0_source_input", self.p0, dtype=np.float32)

            # list of indices into simulation matrix (flattened using MatLab indexing, after adding PML layers)
            write_array_for_kwave(
                f, "sensor_mask_index", self.sensor_indices_flat, dtype=np.uint64
            )

            # HACK
            print("HACK: reduced number of timesteps for testing")
            write_scalar_for_kwave(f, "Nt", self.Nt, dtype=np.uint64)

            # YES, THIS IS INTENTIONAL
            write_scalar_for_kwave(f, "Nx", self.Nz_total, dtype=np.uint64)
            write_scalar_for_kwave(f, "Ny", self.Ny_total, dtype=np.uint64)
            write_scalar_for_kwave(f, "Nz", self.Nx_total, dtype=np.uint64)

            write_scalar_for_kwave(f, "dt", self.dt, dtype=np.float32)
            write_scalar_for_kwave(f, "dx", self.dx, dtype=np.float32)
            write_scalar_for_kwave(f, "dy", self.dy, dtype=np.float32)
            write_scalar_for_kwave(f, "dz", self.dz, dtype=np.float32)

            write_scalar_for_kwave(f, "pml_x_alpha", 2.0, dtype=np.float32)
            write_scalar_for_kwave(f, "pml_y_alpha", 2.0, dtype=np.float32)
            write_scalar_for_kwave(f, "pml_z_alpha", 2.0, dtype=np.float32)

            write_scalar_for_kwave(f, "pml_x_size", self.Npml, dtype=np.uint64)
            write_scalar_for_kwave(f, "pml_y_size", self.Npml, dtype=np.uint64)
            write_scalar_for_kwave(f, "pml_z_size", self.Npml, dtype=np.uint64)

            write_scalar_for_kwave(f, "c_ref", self.c_ref, dtype=np.float32)

            write_scalar_for_kwave(f, "elastic_flag", 0, dtype=np.uint64)
            write_scalar_for_kwave(f, "sensor_mask_type", 0, dtype=np.uint64)
            write_scalar_for_kwave(f, "absorbing_flag", 0, dtype=np.uint64)
            write_scalar_for_kwave(f, "axisymmetric_flag", 0, dtype=np.uint64)
            write_scalar_for_kwave(f, "nonlinear_flag", 0, dtype=np.uint64)
            write_scalar_for_kwave(f, "nonuniform_grid_flag", 0, dtype=np.uint64)
            write_scalar_for_kwave(f, "p0_source_flag", 1, dtype=np.uint64)
            write_scalar_for_kwave(f, "p_source_flag", 0, dtype=np.uint64)
            write_scalar_for_kwave(f, "sxx_source_flag", 0, dtype=np.uint64)
            write_scalar_for_kwave(f, "sxy_source_flag", 0, dtype=np.uint64)
            write_scalar_for_kwave(f, "sxz_source_flag", 0, dtype=np.uint64)
            write_scalar_for_kwave(f, "syy_source_flag", 0, dtype=np.uint64)
            write_scalar_for_kwave(f, "syz_source_flag", 0, dtype=np.uint64)
            write_scalar_for_kwave(f, "szz_source_flag", 0, dtype=np.uint64)
            write_scalar_for_kwave(f, "transducer_source_flag", 0, dtype=np.uint64)
            write_scalar_for_kwave(f, "ux_source_flag", 0, dtype=np.uint64)
            write_scalar_for_kwave(f, "uy_source_flag", 0, dtype=np.uint64)
            write_scalar_for_kwave(f, "uz_source_flag", 0, dtype=np.uint64)

            f.attrs["created_by"] = encode_str("Tim using Python")
            f.attrs["creation_date"] = encode_str(
                datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
            )
            f.attrs["file_description"] = encode_str("okay")

            f.attrs["file_type"] = encode_str("input")
            f.attrs["major_version"] = encode_str("1")
            f.attrs["minor_version"] = encode_str("2")

    def run(self, verbose=False):
        kwave_executable = os.environ.get("KWAVE_EXECUTABLE")

        if kwave_executable is None or not os.path.isfile(kwave_executable):
            raise Exception(
                "Please set the KWAVE_EXECUTABLE environment variable to point to a k-wave executable"
            )

        kwave_temp_folder = os.environ.get("KWAVE_TEMP_FOLDER")
        if kwave_temp_folder is None:
            raise Exception(
                "Please set the KWAVE_TEMP_FOLDER environment variable to point to a writable directory"
            )

        if not os.path.isdir(kwave_temp_folder):
            os.makedirs(kwave_temp_folder)

        hdf5_input_file_path = os.path.join(
            kwave_temp_folder, "temp_kwave_simulation_input.h5"
        )
        hdf5_output_file_path = os.path.join(
            kwave_temp_folder, "temp_kwave_simulation_output.h5"
        )

        if os.path.exists(hdf5_input_file_path):
            os.remove(hdf5_input_file_path)
        if os.path.exists(hdf5_output_file_path):
            os.remove(hdf5_output_file_path)

        assert not os.path.exists(hdf5_input_file_path)
        assert not os.path.exists(hdf5_output_file_path)

        self._write_kwave_input_file(hdf5_input_file_path)

        try:
            res = subprocess.run(
                [
                    kwave_executable,
                    "-i",
                    hdf5_input_file_path,
                    "-o",
                    hdf5_output_file_path,
                ],
                capture_output=(not verbose),
            )
            if res.returncode != 0:
                print("Something went wrong while trying to run kwave:")
                print("STDERR:")
                if res.stderr is not None:
                    print(res.stderr.decode("utf-8"))
                if res.stdout is not None:
                    print("STDOUT:")
                    print(res.stdout.decode("utf-8"))
                exit(-1)
        except subprocess.CalledProcessError as e:
            print("Something went wrong while trying to run kwave:")
            print("STDERR:")
            print(e.stderr.decode("utf-8"))
            print("STDOUT:")
            print(e.stdout.decode("utf-8"))
            exit(-1)

        assert os.path.isfile(hdf5_output_file_path)

        with h5py.File(hdf5_output_file_path, "r") as interpolation_function:
            pressure_vs_time = np.array(interpolation_function["p"])
            assert pressure_vs_time.shape == (
                1,
                self.Nt,
                self.sensor_count,
            )

        signals = pressure_vs_time[0].transpose(1, 0)
        assert signals.shape == (self.sensor_count, self.Nt)

        if self.output_length < self.Nt:
            # print(
            #     f"Low-passing the simulated signal at {self.output_sampling_frequency} Hz before resampling"
            # )
            sos = signal.butter(
                N=10,
                Wn=(0.5 * self.output_sampling_frequency),
                fs=self.simulation_sampling_frequency,
                btype="lowpass",
                output="sos",
            )

            # print(
            #     f"Resampling the low-passed signal to {self.output_sampling_frequency} Hz"
            # )
            signals_lowpassed = signal.sosfilt(sos=sos, x=signals, axis=1)

            assert signals_lowpassed.shape == (
                self.sensor_count,
                self.Nt,
            )

            # This introduces high-frequency artefacts which I do not like one bit
            # signals_resampled = signal.resample(
            #     signals_lowpassed, num=description.output_length, axis=1
            # )

            x_old = np.linspace(0, 1.0, num=self.Nt, endpoint=True)
            interpolation_function = interpolate.interp1d(
                x=x_old, y=signals, kind="linear", axis=1
            )
            x_new = np.linspace(0.0, 1.0, num=self.output_length, endpoint=True)

            signals_resampled = interpolation_function(x_new)

            assert isinstance(signals_resampled, np.ndarray)
            assert signals_resampled.shape == (
                self.sensor_count,
                self.output_length,
            )
            signals = signals_resampled

        os.remove(hdf5_input_file_path)
        os.remove(hdf5_output_file_path)

        return signals.astype(np.float32)
