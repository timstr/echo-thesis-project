from assert_eq import assert_eq
import fix_dead_command_line

import numpy as np

from simulation_description import AcousticMediumProperties, SimulationDescription


def main():
    c_air = 343.0
    c_wood = 4000.0
    c_human = 1540.0
    c_dense_air = c_air * 2.0

    rho_air = 1.225
    rho_wood = 500.0
    rho_human = 1010.0
    rho_dense_air = rho_air * 2.0

    Nx = 8
    Ny = 8
    Nz = 8
    Npml = 10
    spatial_resolution = 1e-2
    dt = 1e-6
    Nt = 64

    air_properties = AcousticMediumProperties(
        speed_of_sound=c_air,  # meters per second
        density=rho_air,  # kilograms per cubic meter
    )
    obstacle_properties = AcousticMediumProperties(
        speed_of_sound=c_dense_air,  # meters per second
        density=rho_dense_air,  # kilograms per cubic meter
        # speed_of_sound=c_wood,  # meters per second
        # density=rho_wood,  # kilograms per cubic meter
    )

    sensor_indices = []
    for i in range(0, Nx, 2):
        for j in range(0, Ny, 2):
            for k in range(0, Nz, 2):
                sensor_indices.append([i, j, k])

    for sensor_index_index, sensor_index in enumerate(sensor_indices):
        desc = SimulationDescription(
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            dx=spatial_resolution,
            dy=spatial_resolution,
            dz=spatial_resolution,
            Npml=Npml,
            dt=dt,
            output_length=Nt,
            Nt=Nt,
            air_properties=air_properties,
            obstacle_properties=obstacle_properties,
            sensor_indices=sensor_indices,
            emitter_indices=(sensor_index[0], sensor_index[1], sensor_index[2]),
        )
        desc.set_obstacles(np.zeros((Nx, Ny, Nz), dtype=np.bool8))
        recordings = desc.run()
        assert_eq(recordings.shape, (len(sensor_indices), Nt))
        max_magnitude_per_receiver = np.amax(recordings, axis=1)
        receiver_with_biggest_magnitude = np.argmax(max_magnitude_per_receiver)
        if receiver_with_biggest_magnitude == sensor_index_index:
            print("Checks out.")
        else:
            print(f"Uh oh! Something is wrong!")
            exit()


if __name__ == "__main__":
    main()
