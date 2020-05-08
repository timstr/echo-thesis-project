import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
import math

from dataset import WaveSimDataset
from field import Field
from featurize import sclog

def main():
    wsds = WaveSimDataset("dataset/v5")

    for i in range(0, 10):
        example = wsds[i]

        obs = example['obstacles']
        echo_raw = example['echo_raw']
        echo_waveshaped = example['echo_waveshaped']
        depthmap = example['depthmap']
        heatmap = example['heatmap']

        # print("List of obstacles:")
        # for o in obs:
        #     print("    ", obs)

        print("Obstacles in field")
        f = Field(512, 512)
        f.add_obstacles(obs)
        plt.imshow(f.get_barrier()[0,0,:,:].cpu())
        plt.show()

        # print("Depthmap (Cartesian projection)")
        # plt.plot(depthmap)
        # plt.show()

        # print("Depthmap (polar projection)")
        # plt.polar(np.linspace(0, math.pi * 2.0, 128), depthmap)
        # plt.show()

        # print("Echo (raw)")
        # plt.plot(echo_raw[0,:])
        # plt.show()

        # print("Echo (waveshaped)")
        # plt.plot(echo_waveshaped[0,:])
        # plt.show()

        print("Obstacle heatmap")
        plt.plot(heatmap)
        plt.show()

if __name__ == "__main__":
    main()