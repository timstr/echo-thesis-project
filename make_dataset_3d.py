from dataset3d import WaveDataset3d
import fix_dead_command_line

import h5py

from current_simulation_description import (
    make_random_obstacles,
    make_simulation_description,
)


def main():
    desc = make_simulation_description()

    dataset = WaveDataset3d(desc, "dataset_v2.h5")

    N = len(dataset)
    M = 10000

    for i in range(N, N + M):
        si = str(i).zfill(4)
        print(f"##############################################################")
        print(f"#                                                            #")
        print(f"#                Creating Dataset Example {si}               #")
        print(f"#                                                            #")
        print(f"##############################################################")

        dataset.simulate_and_append_to_dataset(make_random_obstacles(desc))


if __name__ == "__main__":
    main()
