import fix_dead_command_line

import matplotlib.pyplot as plt

from current_simulation_description import make_simulation_description
from dataset3d import WaveDataset3d
from tof_utils import raymarch_sdf_ground_truth


def main():
    description = make_simulation_description()
    dataset = WaveDataset3d(description, "dataset_v5.h5")

    rm_camera_center = [0.0, 0.0, 0.0]
    rm_camera_up = [0.0, 0.5 * description.Ny * description.dy, 0.0]
    rm_camera_right = [0.0, 0.0, 0.5 * description.Nz * description.dz]
    rm_x_resolution = 1024
    rm_y_resolution = 1024

    img = raymarch_sdf_ground_truth(
        camera_center_xyz=rm_camera_center,
        camera_up_xyz=rm_camera_up,
        camera_right_xyz=rm_camera_right,
        x_resolution=rm_x_resolution,
        y_resolution=rm_y_resolution,
        description=description,
        obstacle_sdf=dataset[0]["sdf"].cuda(),
    )

    plt.imshow(img.cpu().permute(2, 1, 0))
    plt.show()


if __name__ == "__main__":
    main()
