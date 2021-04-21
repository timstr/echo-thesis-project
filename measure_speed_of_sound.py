from wave_field import Field
import matplotlib.pyplot as plt
import numpy as np
import torch
import math


def main():
    size = 2048
    f = Field(size)

    emitter_radius = 1
    emitter_y = 16
    emitter_x = size // 2

    f.get_field()[
        emitter_y - emitter_radius : emitter_y + emitter_radius,
        emitter_x - emitter_radius : emitter_x + emitter_radius,
    ] = 1.0

    plot_interval = size // 10

    time_steps = []
    detected_peaks = []

    plt.ion()
    for i in range(size):
        if (i % plot_interval) == 0:
            plt.cla()
            plt.imshow(f.to_image().cpu())
            plt.gcf().canvas.draw()
            plt.gcf().canvas.flush_events()

        field_below_emitter = f.get_field()[emitter_y:, emitter_x].cpu().numpy()

        peak = np.argmax(field_below_emitter)

        time_steps.append(i)
        detected_peaks.append(peak)

        f.step()

    plt.cla()
    plt.ioff()

    time_steps = np.array(time_steps)
    detected_peaks = np.array(detected_peaks)

    A = np.vstack([time_steps, np.ones(len(time_steps))]).T
    slope, offset = np.linalg.lstsq(A, detected_peaks)[0]
    print(f"slope                      : {slope}")
    print(f"offset                     : {offset}")
    print(f"slope divided by sqrt(3/8) : {slope / math.sqrt(3/8)}")

    plt.plot(time_steps, detected_peaks)

    plt.show()


if __name__ == "__main__":
    main()
