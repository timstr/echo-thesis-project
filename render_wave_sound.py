import torch
import numpy as np
import scipy.io.wavfile as wf

from field import Field, make_random_field
from progress_bar import progress_bar
from featurize import normalize_amplitude

def render_wave_sound(field, receiver_locations, num_samples, rate_hertz, output_path="wave_sound.wav", on_step=None):
    data = []
    for i in range(num_samples):
        if on_step is not None:
            on_step()
        field.step()
        s = []
        for y, x in receiver_locations:
            s.append(field.get_field()[0,0,y,x].item())
        data.append(s)
        if i % 100 == 99 or i == (num_samples - 1):
            progress_bar(i, num_samples)
    data = torch.tensor(data)
    data = normalize_amplitude(data)
    print("\nSaving sound to {}".format(output_path))
    wf.write(output_path, rate_hertz, data.detach().numpy())

def main():
    size = 512
    field = make_random_field(size, size)
    field.get_field()[0,0,size//2,size//2] = 10.0
    loc = [(size//4, size//2 - 10), (size//4, size//2 + 10)]
    sr = 44100
    duration = 2
    render_wave_sound(field, loc, int(sr * duration), sr, "wave_sound.wav")

if __name__ == "__main__":
    main()