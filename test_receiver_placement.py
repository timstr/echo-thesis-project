import fix_dead_command_line

import matplotlib.pyplot as plt
import torch

from current_simulation_description import make_simulation_description
from dataset3d import WaveDataset3d, k_sensor_recordings
from signals_and_geometry import time_of_flight_crop


def main():

    # Questions:
    # - why does grid_sample appear to be normalizing the input grid to [0, num_samples] or similar rather than [-1, 1]???
    # - Are the simulation grid indices correct? Try running a dense simulation and visualizing it. Because of the oblong shape, something will probably be out of bounds if it's wrong

    desc = make_simulation_description()

    dataset = WaveDataset3d(desc, "dataset_v3.h5")

    example = dataset[3]

    recordings = example[k_sensor_recordings]

    crop_size = 128

    for i in range(desc.sensor_count):
        x, y, z = desc.sensor_locations[i]

        audio_cropped = (
            time_of_flight_crop(
                recordings=recordings[[i]].unsqueeze(0),
                sample_locations=torch.Tensor([[[x, y, z]]]),
                emitter_location=torch.Tensor(desc.emitter_location),
                receiver_locations=torch.Tensor(desc.sensor_locations[[i]]),
                speed_of_sound=desc.air_properties.speed_of_sound,
                sampling_frequency=desc.output_sampling_frequency,
                crop_length_samples=crop_size,
            )
            .squeeze(0)
            .squeeze(0)
            .squeeze(0)
        )

        audio_cropped += 0.01 * i

        plt.plot(audio_cropped.cpu().numpy())

    plt.show()


if __name__ == "__main__":
    main()
