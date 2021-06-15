import fix_dead_command_line
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf

from wave_field import Field, make_random_field
from utils import progress_bar


def main():
    size = 512

    # field = Field(size)
    field = make_random_field(size)

    plt.ion()

    mic_location = (size - 8, size // 4)

    freq_exp_begin = -8
    freq_exp_end = -6
    osc_state = 0
    pulse_len = 400
    batch_len = 128
    total_len = 2048

    for use_impulse in [False, True]:
        field.silence()
        t = 0
        mic_recording = []
        pulse_recording = []
        for i in range(total_len // batch_len):
            for j in range(batch_len):
                if use_impulse:
                    s = 1.0 if t == 0 else 0.0
                else:
                    if t < pulse_len:
                        freq_exp_now = freq_exp_begin + (t / pulse_len) * (
                            freq_exp_end - freq_exp_begin
                        )
                        freq_now = math.pow(2.0, freq_exp_now)
                        osc_state += freq_now
                        s = math.sin(osc_state * 2.0 * np.pi)
                        # s = 2.0 * round(0.5 + 0.5 * s) - 1.0
                    else:
                        s = 0.0
                y = size - 8
                x = size // 2
                rad = 1
                field.get_field()[y - rad : y + rad, x - rad : x + rad] = s
                pulse_recording.append(s)

                field.step()
                mic_recording.append(
                    field.get_field()[mic_location[0], mic_location[1]].item()
                )
                t += 1
            progress_bar(t - 1, total_len)

            plt.cla()
            plt.imshow(field.to_image().cpu())
            plt.gcf().canvas.draw()
            plt.gcf().canvas.flush_events()

        mic_recording = np.array(mic_recording)
        pulse_recording = np.array(pulse_recording)
        suffix = "impulse" if use_impulse else "sweep"
        wf.write(f"wave simulation received {suffix}.wav", 44100, mic_recording)
        wf.write(f"wave simulation emitted {suffix}.wav", 44100, pulse_recording)


if __name__ == "__main__":
    main()
