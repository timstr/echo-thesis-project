import os
import sys
import glob

from current_simulation_description import make_simulation_description
from dataset3d import WaveDataset3d, k_sensor_recordings, k_obstacles, k_sdf
from utils import progress_bar


def main(output_path, part_base_path, num_parts, append):
    assert isinstance(output_path, str)
    assert isinstance(part_base_path, str)
    assert isinstance(num_parts, int) and num_parts > 0
    assert isinstance(append, bool)
    desc = make_simulation_description()
    if os.path.exists(output_path):
        if not append:
            print(
                f"Error: attempted to create a new dataset at {output_path} but it already exists"
            )
            exit(-1)
    elif append:
        print(
            f"Error: attempted to append to a dataset at {output_path} but it doesn't exist"
        )
        exit(-1)
    with WaveDataset3d(desc, output_path, write=True) as new_ds:
        for i_part in range(num_parts):
            print(f"Dataset part {i_part+1} of {num_parts}")
            old_path = f"{part_base_path}_{i_part+1}_of_{num_parts}.h5"
            if not os.path.exists(old_path):
                print(f"Error: tried to open the file {old_path} but it doesn't exist")
                exit(-1)
            with WaveDataset3d(desc, old_path) as old_ds:
                N_examples = len(old_ds)
                for i_example, example_dd in enumerate(old_ds):
                    progress_bar(i_example, N_examples)
                    obs = example_dd[k_obstacles]
                    rec = example_dd[k_sensor_recordings]
                    sdf = example_dd[k_sdf]
                    new_ds.append_to_dataset(obstacles=obs, recordings=rec, sdf=sdf)
                print(f"{len(new_ds)} total examples")


def print_usage_and_exit():
    print(
        f"Usage: {sys.argv[0]} combined_output.h5 part_base_path_without_suffix count [--append]"
    )
    exit(-1)


if __name__ == "__main__":
    if len(sys.argv) not in [4, 5]:
        print_usage_and_exit()
    output_path = sys.argv[1]
    part_base_path = sys.argv[2]
    num_parts = int(sys.argv[3])
    append = False
    if len(sys.argv) == 5:
        if sys.argv[4] != "--append":
            print_usage_and_exit()
        append = True
    main(output_path, part_base_path, num_parts, append)
