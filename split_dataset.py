import os
import sys

from current_simulation_description import make_simulation_description
from dataset3d import WaveDataset3d
from utils import progress_bar


def main(input_path, start_index, end_index_inclusive, output_path, append):
    assert isinstance(input_path, str)
    assert isinstance(start_index, int)
    assert isinstance(end_index_inclusive, int)
    assert start_index >= 0
    assert end_index_inclusive >= 0
    assert start_index <= end_index_inclusive
    assert isinstance(output_path, str)
    assert isinstance(append, bool)
    if not os.path.exists(input_path):
        print(f"The file {input_path} does not exist")
        return
    if os.path.exists(output_path):
        if not append:
            print(
                f"Error: attempted to create a new dataset at {output_path}.h5 but it already exists"
            )
            exit(-1)
    elif append:
        print(
            f"Error: attempted to append to a dataset at {output_path} but it doesn't exist"
        )
        exit(-1)
    desc = make_simulation_description()
    with WaveDataset3d(desc, input_path) as input_ds:
        n = len(input_ds)
        if max(start_index, end_index_inclusive) >= n:
            print(
                f"Error: the given indices are out of range. Valid indices are 0 to {n - 1}"
            )
            exit(-1)
        with WaveDataset3d(desc, output_path, write=True) as output_ds:
            for i in range(start_index, end_index_inclusive + 1):
                dd = input_ds[i]
                recordings = dd["sensor_recordings"]
                obstacles = dd["obstacles"]
                sdf = dd["sdf"]
                output_ds.append_to_dataset(
                    obstacles=obstacles, recordings=recordings, sdf=sdf
                )
                progress_bar(i - start_index, end_index_inclusive + 1 - start_index)


def print_usage_and_exit():
    print(
        f"Usage: {sys.argv[0]} --from path/to/dataset.h5 --range start_index end_index_inclusive --to path/to/new_dataset.h5 [--append]"
    )
    exit(-1)


if __name__ == "__main__":
    if len(sys.argv) not in [8, 9]:
        print_usage_and_exit()
    if sys.argv[1] != "--from":
        print_usage_and_exit()
    dataset_path = sys.argv[2]
    if sys.argv[3] != "--range":
        print_usage_and_exit()
    start_index = int(sys.argv[4])
    end_index_inclusive = int(sys.argv[5])
    if sys.argv[6] != "--to":
        print_usage_and_exit()
    output_path = sys.argv[7]
    append = False
    if len(sys.argv) == 9:
        if sys.argv[8] != "--append":
            print_usage_and_exit()
        append = True
    main(dataset_path, start_index, end_index_inclusive, output_path, append)