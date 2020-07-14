import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
import math

from dataset import WaveSimDataset
from field import Field, make_obstacle_map
from featurize import sclog, make_sdf_image_gt

def main():
    wsds = WaveSimDataset(
        "dataset/v8",
        permute=False,
        samples_per_example=1
    )

    # plt.ion()

    for i in range(0, 100):

        example = wsds[i]
        # example = wsds[61]

        # print(i)

        obs = example['obstacles']
        obs_list = example['obstacles_list']
        echo_raw = example['echo_raw']
        echo_waveshaped = example['echo_waveshaped']
        # depthmap = example['depthmap']
        heatmap = example['heatmap']

        # print("List of obstacles:")
        # for o in obs:
        #     print("    ", obs)

        print("Obstacles in field")
        plt.imshow(make_obstacle_map(obs_list, 512, 512).cpu())
        plt.show()

        # print("Obstacles signed distance field")
        # plt.imshow(make_sdf_image_gt({'obstacles_list': obs_list}, 128), vmin=0, vmax=0.5, cmap='hsv')
        # plt.show()

        # print("Depthmap (Cartesian projection)")
        # plt.plot(depthmap)
        # plt.show()

        # print("Depthmap (polar projection)")
        # plt.polar(np.linspace(0, math.pi * 2.0, 128), depthmap)
        # plt.show()

        print("Echo (raw)")
        plt.cla()
        plt.ylim(-1.0, 1.1)
        e = echo_waveshaped
        # e = echo_raw

        num_receivers = e.shape[0]
        for i in range(num_receivers):
            plt.plot(e[i,:])
        # plt.plot(e[0,:])
        plt.show()
        plt.pause(1.0)

        print("Echo (waveshaped)")
        plt.plot(echo_waveshaped[0,:])
        plt.show()

        print("Obstacle heatmap")
        plt.plot(heatmap)
        plt.show()

if __name__ == "__main__":
    main()