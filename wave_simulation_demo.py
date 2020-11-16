import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf

from wave_simulation import step_simulation
from progress_bar import progress_bar

def main():
    h, w = (2048,) * 2

    field_prev = torch.zeros(h, w).cuda()
    field_now = torch.zeros(h, w).cuda()

    blank_half_pad_kernel = torch.zeros(2, 4, 2, 2).cuda()

    plt.ion()

    mic_location = (3 * h // 4, w // 2)
    mic_recording = []

    freq_exp_begin = -8
    freq_exp_end = -4
    osc_state = 0
    pulse_len = 1000
    batch_len = 100
    total_len = 50000

    t = 0
    for i in range(total_len // batch_len):
        for j in range(batch_len):
            if t < pulse_len:
                freq_exp_now = freq_exp_begin + (t / pulse_len) * (freq_exp_end - freq_exp_begin)
                freq_now = math.pow(2.0, freq_exp_now)
                osc_state += freq_now
                s = math.sin(osc_state * 2.0 * np.pi)
                # s = 2.0 * round(0.5 + 0.5 * s) - 1.0
                field_now[h//2-4:h//2+4,w//2-4:w//2+4] = s
            field_now, field_prev = step_simulation(field_now, field_prev, blank_half_pad_kernel)
            mic_recording.append(field_now[mic_location[0], mic_location[1]].item())
            t += 1
        progress_bar(t, total_len)
        
        plt.cla()
        plt.imshow(field_now.cpu(), vmin=-1.0, vmax=1.0)
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()

    mic_recording = np.array(mic_recording)
    wf.write("wave simulation sound.wav", 44100, mic_recording)

if __name__ == "__main__":
    main()
