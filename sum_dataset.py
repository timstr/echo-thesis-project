import os
import torch
from argparse import ArgumentParser
import PIL

import matplotlib
import matplotlib.pyplot as plt

from current_simulation_description import make_simulation_description, minimum_x_units
from dataset3d import WaveDataset3d, k_obstacles
from utils import progress_bar


def plt_screenshot(plt_figure):
    pil_img = PIL.Image.frombytes(
        "RGB", plt_figure.canvas.get_width_height(), plt_figure.canvas.tostring_rgb()
    )
    return pil_img


def main(
    input_paths,
):
    assert isinstance(input_paths, list)
    assert len(input_paths) > 0
    assert all([isinstance(ip, str) for ip in input_paths])

    matplotlib.use("agg")

    for f in input_paths:
        if not os.path.isfile(f):
            print(f"The path {f} does not point to a file")
            exit(-1)
    desc = make_simulation_description()
    for current_input_path in input_paths:
        with WaveDataset3d(desc, current_input_path) as input_ds:
            n = len(input_ds)
            print(f"{current_input_path} ({n} example{'' if n == 1 else 's'})")

            sum_of_obstacles = torch.zeros(
                (desc.Nx, desc.Ny, desc.Nz), dtype=torch.long
            )
            for i in range(n):
                dd = input_ds[i]
                obstacles = dd[k_obstacles]
                sum_of_obstacles.add_(obstacles)

                progress_bar(i, n)

            vmax = torch.max(sum_of_obstacles)

            sum_of_obstacles = sum_of_obstacles / vmax

        slices_x = []
        slices_y = []
        slices_z = []

        num_slices = 10
        for i in range(num_slices):
            x = minimum_x_units + (i * (desc.Nx - minimum_x_units) // num_slices)
            y = i * desc.Ny // num_slices
            z = i * desc.Nz // num_slices

            slices_x.append(sum_of_obstacles[x, :, :])
            slices_y.append(sum_of_obstacles[:, y, :])
            slices_z.append(sum_of_obstacles[:, :, z])

        projection_x = torch.mean(sum_of_obstacles, dim=0)
        projection_y = torch.mean(sum_of_obstacles, dim=1)
        projection_z = torch.mean(sum_of_obstacles, dim=2)

        fig, axes = plt.subplots(
            3, num_slices + 1, figsize=(3 * (num_slices + 1), 9), dpi=80
        )

        def plot_it(arr, i, j):
            axes[i, j].imshow(arr.detach().cpu().permute(1, 0).numpy(), cmap="gray")

        plot_it(projection_x, 0, 0)
        plot_it(projection_y, 1, 0)
        plot_it(projection_z, 2, 0)

        for i in range(num_slices):
            plot_it(slices_x[i], 0, i + 1)
            plot_it(slices_y[i], 1, i + 1)
            plot_it(slices_z[i], 2, i + 1)

        dataset_folder, dataset_name_with_extension = os.path.split(current_input_path)
        dataset_extension = ".h5"
        assert dataset_name_with_extension.endswith(dataset_extension)
        dataset_name = dataset_name_with_extension[: -len(dataset_extension)]

        fig.canvas.draw()

        plt_screenshot(fig).save(f"visualization of {dataset_name}.png")

        print("\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_paths", type=str, metavar="input_path", nargs="+")

    args = parser.parse_args()

    input_paths = args.input_paths

    main(
        input_paths=input_paths,
    )
