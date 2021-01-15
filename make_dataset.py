import fix_dead_command_line
from featurize import all_possible_obstacles, obstacles_occluded
import pickle
import torch
import numpy as np
import math
# import matplotlib.pyplot as plt

from argparse import ArgumentParser

from wave_field import Field
from dataset_config import wavesim_field_size, wavesim_duration, wavesim_emitter_locations, wavesim_receiver_locations
from progress_bar import progress_bar

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--index", dest="index", required=True)
    parser.add_argument("-c", "--count", dest="count", required=True)
    args = parser.parse_args()

    emitter_radius = 2
    num_emitters = len(wavesim_emitter_locations)

    receiver_locations = torch.tensor(wavesim_receiver_locations)
    num_receivers = len(receiver_locations)
    receiver_indices_chunked = receiver_locations.t().chunk(chunks=num_receivers, dim=0)

    all_obstacles = list(all_possible_obstacles(2, 0.05, 0.1, 0.3, 0.1, 0.8, 0.1, 0.9, 3, 4, 4))
    dataset_size = len(all_obstacles)

    # returns list_of_obstacles, sound_buffer
    def make_example(example_index):
        with torch.no_grad():
            print("Creating field")
            field = Field(wavesim_field_size)
            obstacles = all_obstacles[example_index]
            field.add_obstacles(obstacles)
            print("Simulating waves")
            receiver_buf = torch.zeros(num_emitters, num_receivers, wavesim_duration).cuda()
            for i_emitter, (emitter_y, emitter_x) in enumerate(wavesim_emitter_locations):
                field.silence()
                field.get_field()[
                    emitter_y-emitter_radius:emitter_y+emitter_radius,
                    emitter_x-emitter_radius:emitter_x+emitter_radius
                ] = 1.0
                for s in range(wavesim_duration):
                    field.step()
                    receiver_buf[i_emitter,:,s] = field.get_field()[receiver_indices_chunked]
                progress_bar(i_emitter, num_emitters)
            
            max_amp = torch.max(torch.abs(receiver_buf)).item()
            if (max_amp > 1e-6):
                receiver_buf *= 0.5 / max_amp
            sound_buf = receiver_buf.detach().cpu().numpy()
            occlusion = obstacles_occluded(obstacles)
            print(" Done")
            return {
                "obstacles": field.get_obstacles(),
                "impulse_responses": sound_buf,
                "occlusion": occlusion
            }


    output_path = "dataset/v9"

    num_digits = int(math.ceil(math.log10(dataset_size)))

    for i in range(int(args.index), dataset_size, int(args.count)):
        print("Creating example ", i)
        example = make_example(i)
        fname = "{0}/example {1}.pkl".format(output_path, str(i).zfill(num_digits))
        with open(fname, "wb") as outfile:
            pickle.dump(example, outfile)

if __name__ == "__main__":
    main()