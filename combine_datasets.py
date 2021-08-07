import os
import sys
import glob

from current_simulation_description import make_simulation_description
from dataset3d import WaveDataset3d
from utils import progress_bar

dataset_name = "random"


def main(dataset_base_name, num_parts):
    assert isinstance(dataset_base_name, str)
    assert isinstance(num_parts, int) and num_parts > 0
    desc = make_simulation_description()
    new_path = f"/{dataset_base_name}.h5"
    if os.path.exists(new_path):
        print(
            f"Error: attempted to create a new dataset at {dataset_base_name}.h5 but it already exists"
        )
        exit(-1)
    with WaveDataset3d(desc, new_path, write=True) as new_ds:
        for i_part in range(num_parts):
            print(f"Dataset part {i_part+1} of {num_parts}")
            old_path = f"/{dataset_base_name}_{i_part+1}_of_{num_parts}.h5"
            if not os.path.exists(old_path):
                print(f"Error: tried to open the file {old_path} but it doesn't exist")
                exit(-1)
            with WaveDataset3d(desc, old_path) as old_ds:
                N_examples = len(old_ds)
                for i_example, example_dd in enumerate(old_ds):
                    progress_bar(i_example, N_examples)
                    obs = example_dd["obstacles"]
                    rec = example_dd["sensor_recordings"]
                    sdf = example_dd["sdf"]
                    new_ds.append_to_dataset(obstacles=obs, recordings=rec, sdf=sdf)
                print(f"{len(new_ds)} total examples")


def print_usage_and_exit():
    print(f"Usage: {sys.argv[0]} dataset_base_name_without_suffix count")
    exit(-1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print_usage_and_exit()
    dataset_base_name = sys.argv[1]
    num_parts = int(sys.argv[2])
    main(dataset_base_name, num_parts)
