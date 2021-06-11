import h5py
import matplotlib.pyplot as plt
import math
import numpy as np

from kwave_run_simulation import run_kwave_simulation
from kwave_util import make_ball, make_box
from kwave_simulation_description import AcousticMediumProperties, SimulationDescription

c_air = 343.0
c_wood = 4000.0
c_human = 1540.0
c_dense_air = c_air * 2.0

rho_air = 1.225
rho_wood = 500.0
rho_human = 1010.0
rho_dense_air = rho_air * 2.0


def main():
    Nx = 180
    Ny = 60
    Nz = 60
    obstacle_mask = (
        make_ball(Nx, Ny, Nz, Nx - 10, 20, 40, radius=16)
        | make_ball(Nx, Ny, Nz, Nx - 10, 40, 30, radius=8)
        | make_ball(Nx, Ny, Nz, Nx - 10, 25, 12, radius=4)
        | make_ball(Nx, Ny, Nz, Nx - 10, 30, 25, radius=2)
    )
    # obstacle_mask = make_box(
    #     Nx, Ny, Nz, Nx - 10, Ny // 2, Nz // 2, 10, Ny // 2, Nz // 2
    # )
    # obstacle_mask = np.zeros((Nx, Ny, Nz), dtype=np.bool8)

    sensor_center_x = 16
    sensor_center_y = Ny // 2
    sensor_center_z = Nz // 2

    sensor_count_x = 4
    sensor_count_y = 4
    sensor_count_z = 4

    sensor_spacing = 4

    sensor_extent_x = sensor_count_x * (sensor_spacing - 1)
    sensor_extent_y = sensor_count_y * (sensor_spacing - 1)
    sensor_extent_z = sensor_count_z * (sensor_spacing - 1)

    sensor_locations = []

    # Record the entire simulation domain
    # for i in range(Nx):
    #     # for j in range(Ny):
    #     #     for k in range(Nz):
    #     sensor_locations.append((i, Ny // 2, Nz // 2))

    # record a small grid
    for i in range(sensor_count_x):
        for j in range(sensor_count_y):
            for k in range(sensor_count_z):
                x = round(sensor_center_x - (sensor_extent_x / 2) + sensor_spacing * i)
                y = round(sensor_center_y - (sensor_extent_y / 2) + sensor_spacing * j)
                z = round(sensor_center_z - (sensor_extent_z / 2) + sensor_spacing * k)
                sensor_locations.append((x, y, z))

    air_properties = AcousticMediumProperties(
        speed_of_sound=c_air,  # meters per second
        density=rho_air,  # kilograms per cubic meter
    )
    obstacle_properties = AcousticMediumProperties(
        # speed_of_sound=c_dense_air,  # meters per second
        # density=rho_dense_air,  # kilograms per cubic meter
        speed_of_sound=c_wood,  # meters per second
        density=rho_wood,  # kilograms per cubic meter
    )

    spatial_resolution = 1e-2  # meters
    Npml = 10  # spatial count
    dt = 1e-7  # seconds

    sampling_frequency = 96_000.0
    sampling_period = 1.0 / sampling_frequency

    wave_distance_per_time_step = air_properties.speed_of_sound * dt
    corner_to_corner_distance = (
        2.0 * math.sqrt(Nx ** 2 + Ny ** 2 + Nz ** 2) * spatial_resolution
    )
    Nt_original = math.ceil(corner_to_corner_distance / wave_distance_per_time_step)

    Nt_at_sampling_frequency = Nt_original * (dt / sampling_period)
    Nt_at_sampling_frequency_rounded = 2 ** (
        math.ceil(math.log2(Nt_at_sampling_frequency))
    )
    Nt = round(Nt_at_sampling_frequency_rounded * (sampling_period / dt))

    print(
        f"{Nt_original} time steps are required to traverse the simulation twice at a time step of {dt} seconds, at a total duration of {Nt_original * dt} seconds."
    )
    print(
        f"This amounts to {Nt_at_sampling_frequency} samples at {sampling_frequency} Hz, and {Nt_at_sampling_frequency_rounded} samples after rounding to the nearest power of two."
    )
    print(
        f"In order to achieve this, {Nt} time steps are required at the simulation time step."
    )

    desc = SimulationDescription(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        dx=spatial_resolution,
        dy=spatial_resolution,
        dz=spatial_resolution,
        Npml=Npml,
        dt=dt,
        output_length=Nt_at_sampling_frequency_rounded,
        Nt=Nt,
        air_properties=air_properties,
        obstacle_properties=obstacle_properties,
        sensor_locations=sensor_locations,
        impulse_location=(sensor_center_x, sensor_center_y, sensor_center_z),
    )

    desc.set_obstacles(obstacle_mask)

    results = run_kwave_simulation(desc)

    # # temporal delay w.r.t. spatial offset
    # expected_slope = spatial_resolution / (air_properties.speed_of_sound * dt)
    # print(f"expected slope             : {expected_slope}")
    # print(f"expected offset            : {0.0}")

    # positions = np.array(range(50, 150))
    # time_peaks = []
    # for i in range(50, 150):
    #     time_peaks.append(np.argmax(results[:, i]))

    # time_peaks = np.array(time_peaks)
    # assert positions.shape == time_peaks.shape

    # A = np.vstack([positions, np.ones(len(positions))]).T
    # slope, offset = np.linalg.lstsq(A, time_peaks)[0]
    # print(f"slope                      : {slope}")
    # print(f"offset                     : {offset}")

    for i in range(desc.num_sensors):
        plt.plot(results[i])
    plt.show()


if __name__ == "__main__":
    # with h5py.File(C:\\Users\Tim\Documents\soundsimlibs\k-wave\stuff\output_skinny.h5", "r") as f:  # HACK
    #     pressure_vs_time = np.array(f["p"])

    # pressure_vs_time = pressure_vs_time[0]

    # for i in range(pressure_vs_time.shape[1]):
    #     plt.plot(pressure_vs_time[:, i])
    # plt.show()

    main()
