from profiling import Timer
import fix_dead_command_line

import torch
import torchvision
import matplotlib.pyplot as plt

from current_simulation_description import (
    make_simulation_description,
    make_random_obstacles,
)
from tof_utils import obstacle_map_to_sdf, colourize_sdf
from utils import assert_eq


def slice_and_plot(tensor, description, plt_axis, colour_fn=None):
    tensor = tensor.float().detach()
    if colour_fn is None:
        vmin = torch.min(tensor)
        vmax = torch.max(tensor)
        tensor = (tensor - vmin) / (vmax - vmin)
    num_slices = 10
    slices = []
    for i in range(num_slices):
        z = i * description.Nz // num_slices
        the_slice = tensor[:, :, z]
        if colour_fn is not None:
            the_slice = colour_fn(the_slice)
            assert_eq(the_slice.shape, (3, description.Nx, description.Ny))
        else:
            the_slice = the_slice.unsqueeze(0)
        slices.append(the_slice)

    img_grid = torchvision.utils.make_grid(
        tensor=slices, nrow=5, pad_value=0.5
    ).permute(2, 1, 0)
    plt_axis.imshow(img_grid.cpu().numpy())
    plt_axis.axis("off")


def main():
    timer_0 = Timer("setup")
    desc = make_simulation_description()
    desc.set_obstacles(make_random_obstacles(desc))
    obs = torch.tensor(desc.obstacle_mask).cuda()
    timer_0.done()

    sdf = obstacle_map_to_sdf(obs, desc)

    timer_1 = Timer("Plotting")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=80)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))

    slice_and_plot(obs, desc, ax[0])
    slice_and_plot(sdf, desc, ax[1], colour_fn=colourize_sdf)
    timer_1.done()

    plt.show()


if __name__ == "__main__":
    main()
