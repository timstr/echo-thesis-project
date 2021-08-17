import fix_dead_command_line

import math
import matplotlib.pyplot as plt
import torch

from current_simulation_description import make_simulation_description
from dataset3d import WaveDataset3d
from tof_utils import convolve_recordings, make_fm_chirp, time_of_flight_crop
from time_of_flight_net import sclog


def main():

    # Questions:
    # - why does grid_sample appear to be normalizing the input grid to [0, num_samples] or similar rather than [-1, 1]???
    # - Are the simulation grid indices correct? Try running a dense simulation and visualizing it. Because of the oblong shape, something will probably be out of bounds if it's wrong

    desc = make_simulation_description()

    dataset = WaveDataset3d(desc, "dataset_train.h5")

    example = dataset[0]

    obstacles = example["obstacles"]

    z_index = desc.Nz // 2

    # obstacles_z_slice = obstacles[:, :, z_index]

    obstacles_depthmap = torch.zeros(desc.Nx, desc.Ny)
    for z in range(desc.Nz):
        obstacles_depthmap[obstacles[:, :, z]] = z / (desc.Nz - 1)

    recordings = example["sensor_recordings"]

    fm_chirp = torch.tensor(
        make_fm_chirp(
            begin_frequency_Hz=16_000.0,
            end_frequency_Hz=0_000.0,
            sampling_frequency=desc.output_sampling_frequency,
            chirp_length_samples=math.ceil(0.0005 * desc.output_sampling_frequency),
        )
    ).float()

    recordings = convolve_recordings(fm_chirp, recordings, desc)

    plt.ion()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=80)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))

    crop_size = 128

    ax_l = ax[0]
    ax_r = ax[1]

    x_index = desc.emitter_indices[0]
    y_index = desc.emitter_indices[1]

    dragging = False

    stay_open = True

    def onclick(event):
        nonlocal x_index
        nonlocal y_index
        nonlocal dragging
        if event.xdata is not None and event.ydata is not None:
            x_index = event.xdata
            y_index = event.ydata
        dragging = True

    def onrelease(event):
        nonlocal dragging
        dragging = False

    def onmove(event):
        nonlocal x_index
        nonlocal y_index
        if not dragging:
            return
        if event.xdata is not None and event.ydata is not None:
            x_index = event.xdata
            y_index = event.ydata

    def on_close(event):
        nonlocal stay_open
        stay_open = False

    cid_click = fig.canvas.mpl_connect("button_press_event", onclick)
    cid_release = fig.canvas.mpl_connect("button_release_event", onrelease)
    cid_move = fig.canvas.mpl_connect("motion_notify_event", onmove)
    cid_close = fig.canvas.mpl_connect("close_event", on_close)

    while stay_open:
        ax_l.cla()
        ax_r.cla()

        x_coord = (x_index - desc.emitter_indices[0]) * desc.dx
        y_coord = (y_index - desc.emitter_indices[1]) * desc.dy
        z_coord = (z_index - desc.emitter_indices[2]) * desc.dz

        audio_cropped = sclog(
            time_of_flight_crop(
                recordings=recordings.unsqueeze(0),
                sample_locations=torch.Tensor([[[x_coord, y_coord, z_coord]]]),
                emitter_location=torch.Tensor(desc.emitter_location),
                receiver_locations=torch.Tensor(desc.sensor_locations),
                speed_of_sound=desc.air_properties.speed_of_sound,
                sampling_frequency=desc.output_sampling_frequency,
                crop_length_samples=crop_size,
            )
            .squeeze(0)
            .squeeze(0)
        )

        # ax_l.set_ylim(-5e-4, 5e-4)
        ax_l.set_ylim(-1, 1)
        ax_l.set_xlim(0, crop_size)

        for j in range(desc.sensor_count):
            ax_l.plot(audio_cropped[j].detach())

        ax_r.set_xlim(0, desc.Nx)
        ax_r.set_ylim(0, desc.Ny)
        # ax_r.imshow(obstacles_z_slice.permute(1, 0))
        ax_r.imshow(obstacles_depthmap.permute(1, 0))

        marker_x = x_index
        marker_y = y_index

        marker_size = 5
        marker_thickness = 2.0
        ax_r.plot(
            [marker_x - marker_size, marker_x + marker_size],
            [marker_y - marker_size, marker_y + marker_size],
            c="red",
            linewidth=marker_thickness,
        )
        ax_r.plot(
            [marker_x - marker_size, marker_x + marker_size],
            [marker_y + marker_size, marker_y - marker_size],
            c="red",
            linewidth=marker_thickness,
        )
        ax_r.scatter(desc.emitter_indices[0:1], desc.emitter_indices[1:2], marker="$e$")
        ax_r.scatter(desc.sensor_indices[:, 0], desc.sensor_indices[:, 1], marker="$r$")

        fig.canvas.draw()
        fig.canvas.flush_events()


if __name__ == "__main__":
    main()
