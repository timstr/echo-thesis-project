import fix_dead_command_line

import numpy as np
from h5ds import H5DS
import h5py

import torch
import math
import matplotlib.pyplot as plt

from plot_utils import plt_screenshot
from train_time_of_flight_net import restore_module
from current_simulation_description import make_simulation_description
from dataset3d import WaveDataset3d
from time_of_flight_net import TimeOfFlightNet
from tof_utils import (
    obstacle_map_to_sdf,
    raymarch_sdf_ground_truth,
    raymarch_sdf_prediction,
    vector_cross,
    vector_length,
    vector_normalize,
)


def main():
    description = make_simulation_description()
    dataset = WaveDataset3d(description, "dataset_v5.h5")

    # rm_camera_center = [0.0, 0.0, 0.0]
    # rm_camera_up = [0.0, 0.5 * description.Ny * description.dy, 0.0]
    # rm_camera_right = [0.0, 0.0, 0.5 * description.Nz * description.dz]

    rm_camera_center = [-0.2, -0.4, 1.0]
    rm_camera_up = vector_normalize([-0.2, 1.0, 0.2], norm=0.5)
    rm_camera_right = vector_normalize([1.0, 0.0, 1.0], norm=1.0)
    rm_x_resolution = 1024
    rm_y_resolution = 512

    # Make sure that camera directions are orthogonal
    assert (
        abs(
            1.0
            - vector_length(
                vector_cross(
                    vector_normalize(rm_camera_up), vector_normalize(rm_camera_right)
                )
            )
        )
        < 1e-6
    )

    plt.ion()

    for i in range(0, 256):
        obstacle_map = torch.zeros(
            (description.Nx, description.Ny, description.Nz),
            dtype=torch.bool,
            device="cuda",
        )

        # obstacle_map[10:-10, 10:-10, 10:-10] = 1.0

        echo4ch_obstacles_h5file = h5py.File("echo4ch_obstacles.h5", "r")
        echo4ch_obstacle_ds = H5DS(
            name="obstacles", dtype=np.bool8, shape=(64, 64, 64), extensible=True
        )
        assert echo4ch_obstacle_ds.exists(echo4ch_obstacles_h5file)
        echo4ch_obstacle = echo4ch_obstacle_ds.read(echo4ch_obstacles_h5file, i)
        obstacle_map[-64:, :, :] = torch.tensor(echo4ch_obstacle[:, 2:-2, 2:-2]).cuda()

        obstacle_sdf = obstacle_map_to_sdf(obstacle_map, description)

        img = raymarch_sdf_ground_truth(
            camera_center_xyz=rm_camera_center,
            camera_up_xyz=rm_camera_up,
            camera_right_xyz=rm_camera_right,
            x_resolution=rm_x_resolution,
            y_resolution=rm_y_resolution,
            description=description,
            # obstacle_sdf=dataset[0]["sdf"].cuda()
            obstacle_sdf=obstacle_sdf,
        )

        # sensor_indices = range(0, 64, 1)
        # model = TimeOfFlightNet(
        #     speed_of_sound=description.air_properties.speed_of_sound,
        #     sampling_frequency=description.output_sampling_frequency,
        #     recording_length_samples=description.output_length,
        #     crop_length_samples=128,
        #     emitter_location=description.emitter_location,
        #     receiver_locations=description.sensor_locations[sensor_indices],
        # ).to("cuda")
        # restore_module(model, "models/model_5367.dat")

        # img = raymarch_sdf_prediction(
        #     camera_center_xyz=rm_camera_center,
        #     camera_up_xyz=rm_camera_up,
        #     camera_right_xyz=rm_camera_right,
        #     x_resolution=rm_x_resolution,
        #     y_resolution=rm_y_resolution,
        #     description=description,
        #     model=model,
        #     recordings=dataset[0]["sensor_recordings"].cuda(),
        # )

        plt.cla()
        plt.imshow(img.cpu().permute(2, 1, 0))

        # plt.show()
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()

        plt_screenshot(plt.gcf()).save(f"temp_img/img_{str(i).zfill(3)}.png")


if __name__ == "__main__":
    main()
