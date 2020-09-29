import numpy as np
import matplotlib.pyplot as plt
import noise
import math

from featurize import shortest_distance_to_obstacles, make_random_obstacles, distance_along_line_of_sight

def clip01(x, threshold=1e-3):
    output = np.zeros(x.shape, dtype=float)
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            output[i,j] = 0.0 if x[i,j] < threshold else 1.0
    return output

def main():    
    size = 256

    obs = make_random_obstacles()

    sdf_orig = np.zeros((size, size), dtype=float)

    for i, y in enumerate(np.linspace(0.0, 1.0, size)):
        for j, x in enumerate(np.linspace(0.0, 1.0, size)):
            dist = shortest_distance_to_obstacles(obs, y, x)
            sdf_orig[i,j] = dist

    hm_orig = clip01(sdf_orig)


    plt.imshow(hm_orig, vmin=0.0, vmax=1.0)
    plt.gcf().suptitle("Heatmap (original)")
    plt.show()
    plt.imshow(sdf_orig)#, vmin=0, vmax=0.5, cmap='hsv')
    plt.gcf().suptitle("Signed Distance Field (original)")
    plt.show()

    sdf_noisy = np.zeros((size, size), dtype=float)
    for i, y in enumerate(np.linspace(0.0, 1.0, size)):
        for j, x in enumerate(np.linspace(0.0, 1.0, size)):
            sdf_noisy[i,j] = sdf_orig[i,j] + 0.25 * noise.pnoise2(x * 5.0, y * 5.0)

    plt.imshow(sdf_noisy, vmin=0, vmax=0.5, cmap='hsv')
    plt.gcf().suptitle("Signed Distance Field (noisy)")
    plt.show()

    hm_from_sdf_noise = clip01(sdf_noisy)
    plt.imshow(hm_from_sdf_noise, vmin=0.0, vmax=1.0)
    plt.gcf().suptitle("Heatmap (clipped from noisy SDF)")
    plt.show()

if __name__ == "__main__":
    main()
