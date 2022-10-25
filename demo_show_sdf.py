import fix_dead_command_line
import cleanup_when_killed

import os
import torch
import math
from argparse import ArgumentParser
import PIL

from visualization import (
    colourize_sdf,
)
from signals_and_geometry import (
    sample_obstacle_map,
)
from split_till_it_fits import SplitSize
from assert_eq import assert_eq
from which_device import get_compute_device
from current_simulation_description import (
    make_simulation_description,
)
from dataset3d import WaveDataset3d, k_sdf


def main():
    parser = ArgumentParser()
    parser.add_argument("path_to_dataset", type=str)
    parser.add_argument("indices", nargs="*", metavar="indices", type=int, default=0)
    parser.add_argument("--outputpath", type=str, dest="outputpath", required=True)
    args = parser.parse_args()

    description = make_simulation_description()

    dataset = WaveDataset3d(description, args.path_to_dataset)

    x_ls = torch.linspace(
        start=description.xmin,
        end=description.xmax,
        steps=description.Nx,
        device=get_compute_device(),
    )
    y_ls = torch.linspace(
        start=description.ymin,
        end=description.ymax,
        steps=description.Ny,
        device=get_compute_device(),
    )

    x_grid, y_grid = torch.meshgrid([x_ls, y_ls])

    z_grid = 0.5 * (description.zmin + description.zmax) * torch.ones_like(x_grid)
    xyz = torch.stack([x_grid, y_grid, z_grid], dim=2).to(get_compute_device())

    assert_eq(xyz.shape, (description.Nx, description.Ny, 3))
    xyz = xyz.reshape((description.Nx * description.Ny), 3)

    for index in args.indices:
        print(f"Rendering example {index}")
        example = dataset[index].to(get_compute_device())

        obstacle_map = example[k_sdf]

        sdf = sample_obstacle_map(
            obstacle_map.unsqueeze(0), xyz.unsqueeze(0), description
        ).squeeze(0)

        assert_eq(sdf.shape, (description.Nx * description.Ny,))
        sdf = sdf.reshape(description.Nx, description.Ny)

        # img = colourize_sdf(sdf)
        sdf = (sdf <= 0.0).float()
        img = sdf.unsqueeze(0).repeat(3, 1, 1)
        img = 0.9 - 0.5 * img

        dataset_folder, dataset_name = os.path.split(args.path_to_dataset)
        assert dataset_name.endswith(".h5")
        dataset_name = dataset_name[: -len(".h5")]
        dataset_size = len(dataset)
        num_digits = math.ceil(math.log10(dataset_size))
        index_str = str(index).zfill(num_digits)

        if not os.path.exists(args.outputpath):
            os.makedirs(args.outputpath)

        filename = f"img_sdf_{dataset_name}_{index_str}.png"
        filepath = os.path.join(args.outputpath, filename)

        img = torch.clamp(img * 255, min=0.0, max=255.0).to(torch.uint8)

        PIL.Image.fromarray(img.permute(2, 1, 0).cpu().numpy()).save(filepath)


if __name__ == "__main__":
    with torch.no_grad():
        main()
