import pickle
import torch
import numpy as np
import math
# import matplotlib.pyplot as plt

from argparse import ArgumentParser

from field import Field, make_random_field
from progress_bar import progress_bar

def main():
    dataset_size = 4096

    parser = ArgumentParser()
    parser.add_argument("-i", "--index", dest="index", required=True)
    parser.add_argument("-c", "--count", dest="count", required=True)
    args = parser.parse_args()

    field_size = 512
    sim_len = 16384
    steps_per_sample = 4
    num_samples = sim_len // steps_per_sample

    # receiver_locations = [
    #     (field_size//2 - 10, field_size//2 - 10),
    #     (field_size//2 - 10, field_size//2 + 10),
    #     (field_size//2 + 10, field_size//2 - 10),
    #     (field_size//2 + 10, field_size//2 + 10),
    # ]

    xx, yy = np.meshgrid(
        np.linspace(0.9, 0.98, 4),
        np.linspace(0.15, 0.85, 16)
    )
    receiver_locations = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
    receiver_indices = np.round(field_size * receiver_locations)
    receiver_indices = torch.tensor(receiver_indices, dtype=torch.long).cuda()
    num_receivers = receiver_indices.shape[0]
    receiver_indices_chunked = receiver_indices.t().chunk(chunks=num_receivers, dim=0)

    speaker_width = 128
    speaker_amplitude = torch.cos(torch.linspace(-0.5, 0.5, speaker_width) * math.pi).cuda()

    cycle_length = 200
    num_cycles = 8
    signal_len = num_cycles * cycle_length
    signal_sine = torch.sin(torch.linspace(0.0, num_cycles * 2.0 * math.pi, signal_len))
    signal_square_0_1 = torch.round(0.5 + 0.5 * signal_sine)
    signal_square = 2.0 * signal_square_0_1 - 1.0
    signal = signal_square.cuda()

    # returns list_of_obstacles, sound_buffer
    def make_example(example_index):
        with torch.no_grad():
            print("Creating field")
            field = make_random_field(field_size, field_size)
            # field = Field(field_size, field_size)
            # f = example_index / dataset_size
            # if (f < 0.25):
            #     t = f * 4.0
            #     # move from 0.2,0.2 to 0.8,0.2
            #     x = 0.2 + 0.6 * t
            #     y = 0.2
            # elif (f < 0.5):
            #     t = (f - 0.25) * 4.0
            #     # move from 0.8,0.2 to 0.8,0.8
            #     x = 0.8
            #     y = 0.2 + 0.6 * t
            # elif (f < 0.75):
            #     t = (f - 0.5) * 4.0
            #     # move from 0.8,0.8 to 0.2,0.8
            #     x = 0.8 - 0.6 * t
            #     y = 0.8
            # elif (f <= 1.0):
            #     t = (f - 0.75) * 4.0
            #     # move from 0.2,0.8 to 0.2,0.2
            #     x = 0.2
            #     y = 0.8 - 0.6 * t
            # field.add_rectangle(y, x, 0.1, 0.1)
            print("Simulating waves")
            receiver_buf = torch.zeros(num_receivers, num_samples).cuda()
            for i in range(num_samples):
                for j in range(steps_per_sample):
                    time_idx = i * steps_per_sample + j
                    if (time_idx < signal_len):
                        signal_val = signal[time_idx]
                        field.get_field()[
                            0,
                            0,
                            field_size-8:field_size-4,
                            (field_size//2 - speaker_width//2) : (field_size//2 + speaker_width//2)
                        ] = signal_val * speaker_amplitude.unsqueeze(0)
                    field.step()
                field_amp = field.get_field()[0,0,:,:]
                receiver_buf[:,i] = field_amp[receiver_indices_chunked]
                progress_bar(i, num_samples)
            
            max_amp = torch.max(torch.abs(receiver_buf)).item()
            if (max_amp > 1e-3):
                receiver_buf *= 0.5 / max_amp
            sound_buf = receiver_buf.t().detach().cpu().numpy()
            print(" Done")
            return field.get_obstacles(), sound_buf


    output_path = "dataset/v8"

    num_digits = int(math.ceil(math.log10(dataset_size)))

    for i in range(int(args.index), dataset_size, int(args.count)):
        print("Creating example ", i)
        example = make_example(i)
        fname = "{0}/example {1}.pkl".format(output_path, str(i).zfill(num_digits))
        with open(fname, "wb") as outfile:
            pickle.dump(example, outfile)

if __name__ == "__main__":
    main()