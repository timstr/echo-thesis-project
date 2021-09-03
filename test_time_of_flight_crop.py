import fix_dead_command_line

import sys
import math
import matplotlib.pyplot as plt
import torch

from current_simulation_description import make_simulation_description
from dataset3d import WaveDataset3d, k_sensor_recordings, k_obstacles
from signals_and_geometry import convolve_recordings, make_fm_chirp, time_of_flight_crop


def main():
    if len(sys.argv) not in [2, 3]:
        print(f"usage: {sys.argv[0]} path/to/dataset.h5 [index]")
        exit(-1)
    path_to_dataset = sys.argv[1]
    dataset_index = 0
    if len(sys.argv) == 3:
        dataset_index = int(sys.argv[2])

    description = make_simulation_description()

    # dataset = WaveDataset3d(desc, "dataset_train.h5")
    dataset = WaveDataset3d(description, path_to_dataset)

    example = dataset[dataset_index]

    obstacles = example[k_obstacles]

    z_index = description.Nz // 2

    # obstacles_z_slice = obstacles[:, :, z_index]

    obstacles_depthmap = torch.zeros(description.Nx, description.Ny)
    for z in range(description.Nz):
        obstacles_depthmap[obstacles[:, :, z]] = z / (description.Nz - 1)

    recordings = example[k_sensor_recordings]

    fm_chirp = torch.tensor(
        make_fm_chirp(
            begin_frequency_Hz=32_000.0,
            end_frequency_Hz=0_000.0,
            sampling_frequency=description.output_sampling_frequency,
            chirp_length_samples=math.ceil(
                0.001 * description.output_sampling_frequency
            ),
            wave="sine",
        )
    ).float()

    recordings = convolve_recordings(fm_chirp, recordings, description)

    plt.ion()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=80)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))

    crop_size = 128

    ax_l = ax[0]
    ax_r = ax[1]

    x_index = description.emitter_indices[0]
    y_index = description.emitter_indices[1]

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

        x_coord = (x_index - description.emitter_indices[0]) * description.dx
        y_coord = (y_index - description.emitter_indices[1]) * description.dy
        z_coord = (z_index - description.emitter_indices[2]) * description.dz

        audio_cropped = (
            time_of_flight_crop(
                recordings=recordings.unsqueeze(0),
                sample_locations=torch.Tensor([[[x_coord, y_coord, z_coord]]]),
                emitter_location=torch.Tensor(description.emitter_location),
                receiver_locations=torch.Tensor(description.sensor_locations),
                speed_of_sound=description.air_properties.speed_of_sound,
                sampling_frequency=description.output_sampling_frequency,
                crop_length_samples=crop_size,
                apply_amplitude_correction=True,
            )
            .squeeze(0)
            .squeeze(0)
        )

        # ax_l.set_ylim(-5e-4, 5e-4)
        ax_l.set_ylim(-1, 1)
        ax_l.set_xlim(0, crop_size)

        for j in range(description.sensor_count):
            ax_l.plot(audio_cropped[j].detach())

        ax_r.set_xlim(0, description.Nx)
        ax_r.set_ylim(0, description.Ny)
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
        ax_r.scatter(
            description.emitter_indices[0:1],
            description.emitter_indices[1:2],
            marker="$e$",
        )
        ax_r.scatter(
            description.sensor_indices[:, 0],
            description.sensor_indices[:, 1],
            marker="$r$",
        )

        fig.canvas.draw()
        fig.canvas.flush_events()


if __name__ == "__main__":
    main()
