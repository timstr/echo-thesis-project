import fix_dead_command_line

from featurize_audio import crop_audio_from_location
from config import EmitterConfig, InputConfig, OutputConfig, ReceiverConfig, TrainingConfig
import matplotlib.pyplot as plt
import torch

from dataset import WaveSimDataset
from featurize import make_sdf_image_gt, colourize_sdf


def main():

    tcfg = TrainingConfig(max_examples=512)
    ecfg = EmitterConfig()
    rcfg = ReceiverConfig()
    # icfg = InputConfig(ecfg, rcfg, format="audioraw", tof_crop_size=256)
    icfg = InputConfig(ecfg, rcfg, format="gcc", tof_crop_size=256)
    ocfg = OutputConfig(format="sdf")

    wsds = WaveSimDataset(tcfg, icfg, ocfg, ecfg, rcfg)

    plt.ion()

    # example = wsds[120]
    example = wsds[500]

    obs_list = example['obstacles_list']
    audio = torch.tensor(example['input'])

    dummy_batch = {"obstacles_list": [obs_list]}

    res = 256

    sdf_image = colourize_sdf(make_sdf_image_gt(dummy_batch, res)).cpu()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=80)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))

    ax_l = ax[0]
    ax_r = ax[1]

    sample_x = 0.5
    sample_y = 0.5

    dragging = False

    stay_open = True

    def onclick(event):
        nonlocal sample_x
        nonlocal sample_y
        nonlocal dragging
        if event.xdata is not None and event.ydata is not None:
            sample_x = event.xdata / res
            sample_y = event.ydata / res
        dragging = True

    def onrelease(event):
        nonlocal dragging
        dragging = False

    def onmove(event):
        nonlocal sample_x
        nonlocal sample_y
        if not dragging:
            return
        if event.xdata is not None and event.ydata is not None:
            sample_x = event.xdata / res
            sample_y = event.ydata / res

    def on_close(event):
        nonlocal stay_open
        stay_open = False

    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_release = fig.canvas.mpl_connect('button_release_event', onrelease)
    cid_move = fig.canvas.mpl_connect('motion_notify_event', onmove)
    cid_close = fig.canvas.mpl_connect('close_event', on_close)

    while stay_open:
        ax_l.cla()
        ax_r.cla()

        audio_cropped = crop_audio_from_location(
            audio, icfg, sample_y, sample_x)

        ax_l.set_ylim(0.0, 1.0)
        ax_l.set_xlim(0, icfg.tof_crop_size)

        for j in range(icfg.num_channels):
            top = j / icfg.num_channels
            bottom = (j + 1) / icfg.num_channels
            center = (top + bottom) * 0.5
            scale = bottom - top
            ax_l.plot(center + scale * audio_cropped[j].detach())

        ax_r.set_xlim(0, res)
        ax_r.set_ylim(res, 0)
        ax_r.imshow(sdf_image)

        marker_x = sample_x * res
        marker_y = sample_y * res

        marker_size = res * 0.02
        marker_thickness = 2.0
        ax_r.plot(
            [marker_x - marker_size, marker_x + marker_size],
            [marker_y - marker_size, marker_y + marker_size],
            c="red",
            linewidth=marker_thickness
        )
        ax_r.plot(
            [marker_x - marker_size, marker_x + marker_size],
            [marker_y + marker_size, marker_y - marker_size],
            c="red",
            linewidth=marker_thickness
        )

        fig.canvas.draw()
        fig.canvas.flush_events()


if __name__ == "__main__":
    main()
