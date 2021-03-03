import fix_dead_command_line

from config import EmitterConfig, InputConfig, OutputConfig, ReceiverConfig, TrainingConfig
import matplotlib.pyplot as plt
import numpy as np

from dataset import WaveSimDataset
from featurize import make_depthmap_gt, make_heatmap_image_gt, make_sdf_image_gt, obstacles_occluded, red_white_blue_banded

def main():
    tcfg = TrainingConfig()#max_examples=128)
    ecfg = EmitterConfig()
    rcfg = ReceiverConfig()
    icfg = InputConfig(ecfg, rcfg, format="spectrogram")
    ocfg = OutputConfig(format="sdf")

    wsds = WaveSimDataset(tcfg, icfg, ocfg, ecfg, rcfg)

    animate = True

    if animate:
        plt.ion()

    # for i in range(6166, 7331):
    for i in range(len(wsds)):

        example = wsds[i*8]
        # example = wsds[61]

        obs_list = example['obstacles_list']
        spectrograms = example['input']
        sdf = example["output"]

        dummy_batch = { "obstacles_list": [obs_list]}

        print(f"Obstacle {i}")

        print("Occlusion:", obstacles_occluded(obs_list))

        # print("Obstacles in field")
        plt.cla()
        plt.imshow(make_heatmap_image_gt(dummy_batch, 512).cpu())
        # plt.show()

        
        # print("Obstacles depthmap")
        # plt.cla()
        plt.plot(512 * (1.0 - make_depthmap_gt(dummy_batch, 512).cpu()))
        plt.gca().set_ylim([512.0, 0.0])
        # plt.show()

        if animate:
            plt.gcf().canvas.flush_events()
            plt.gcf().canvas.draw()
            continue
        else:
            plt.show()

        # print("Obstacles signed distance field")
        plt.cla()
        plt.imshow(red_white_blue_banded(make_sdf_image_gt(dummy_batch, 512)).cpu())
        plt.show()

        # print("Impulse responses (waveforms)")
        # colours = [
        #     (1.0, 0.0, 0.0),
        #     (0.0, 1.0, 0.0),
        #     (0.0, 0.0, 1.0),
        #     (0.5, 0.0, 0.0),
        #     (0.0, 0.5, 0.0),
        #     (0.0, 0.0, 0.5),
        #     (0.5, 0.5, 0.0),
        #     (0.5, 0.0, 0.5),
        #     (0.0, 0.5, 0.5),
        # ]
        # fig, ax = plt.subplots(n_emitters, 1)
        # for e in range(n_emitters):
        #     ax[e].set_ylim(-0.1, 0.25)
        #     for r in range(n_receivers):
        #         ax[e].plot(impulse_responses[e,r], c=colours[r])
        # plt.show()

        
        print("Impulse responses (spectrograms)")
        print("spectrogram.shape =", spectrograms.shape)
        vmin = np.min(spectrograms)
        vmax = np.max(spectrograms)
        print(f"Min = {vmin}")
        print(f"Max = {vmax}")
        n = spectrograms.shape[0]
        fig, ax = plt.subplots(n)
        for i in range(n):
                ax[i].imshow(spectrograms[i], vmin=vmin, vmax=vmax, cmap="gray")
        plt.show()


if __name__ == "__main__":
    main()