from genericpath import isfile
import os
import sys
from argparse import ArgumentParser
import glob

from current_simulation_description import make_simulation_description
from dataset3d import WaveDataset3d
from utils import progress_bar
from tof_utils import obstacle_map_to_sdf


def main(
    input_paths,
    start_index,
    end_index_inclusive,
    output_path,
    append,
    recompute_sdf,
    skip_duplicates,
):
    assert isinstance(input_paths, list)
    assert len(input_paths) > 0
    assert all([isinstance(ip, str) for ip in input_paths])
    assert isinstance(start_index, int) or start_index is None
    assert isinstance(end_index_inclusive, int) or start_index is None
    assert (start_index is None) == (end_index_inclusive is None)
    assert start_index is None or start_index >= 0
    assert end_index_inclusive is None or end_index_inclusive >= 0
    assert start_index is None or start_index <= end_index_inclusive
    assert isinstance(output_path, str)
    assert isinstance(append, bool)
    assert isinstance(recompute_sdf, bool)
    assert isinstance(skip_duplicates, bool)

    for f in input_paths:
        if not os.path.isfile(f):
            print(f"The path {f} does not point to a file")
            exit(-1)
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
    desc = make_simulation_description()
    with WaveDataset3d(desc, output_path, write=True) as output_ds:
        for current_input_path in input_paths:
            num_duplicates = 0
            with WaveDataset3d(desc, current_input_path) as input_ds:
                n = len(input_ds)
                if start_index is not None:
                    if max(start_index, end_index_inclusive) >= n:
                        print(
                            f"Error: the given indices are out of range. Valid indices are 0 to {n - 1}"
                        )
                        exit(-1)
                    current_start_index = start_index
                    current_end_index_inclusive = end_index_inclusive
                else:
                    current_start_index = 0
                    current_end_index_inclusive = n - 1
                n_out = len(output_ds)
                print(
                    f"{current_input_path} ({n} example{'' if n == 1 else 's'}) ==> {output_path} ({n_out} example{'' if n_out == 1 else 's'})"
                )
                for i in range(current_start_index, current_end_index_inclusive + 1):
                    dd = input_ds[i]
                    recordings = dd["sensor_recordings"]
                    obstacles = dd["obstacles"]
                    if recompute_sdf:
                        sdf = obstacle_map_to_sdf(obstacles.cuda(), desc).cpu()
                    else:
                        sdf = dd["sdf"]
                    was_added = output_ds.append_to_dataset(
                        obstacles=obstacles,
                        recordings=recordings,
                        sdf=sdf,
                        skip_duplicates=skip_duplicates,
                    )
                    if not was_added:
                        num_duplicates += 1
                    progress_bar(
                        i - current_start_index,
                        current_end_index_inclusive + 1 - current_start_index,
                    )

            print(
                f"{num_duplicates} duplicate{' was' if num_duplicates == 1 else 's were'} skipped."
            )
        n_out = len(output_ds)
        print(f"{output_path} now contains {n_out} example{'' if n_out == 1 else 's'}")


def print_usage_and_exit():
    print(
        f"Usage: {sys.argv[0]} --from path/to/dataset.h5 --range start_index end_index_inclusive --to path/to/new_dataset.h5 [--append] [--recomputesdf] [--skipduplicates]"
    )
    exit(-1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_paths", type=str, metavar="input_path", nargs="+")
    parser.add_argument("--dst", type=str, dest="dst", required=True)
    parser.add_argument("--start", type=int, dest="start", required=False, default=None)
    parser.add_argument("--end", type=int, dest="end", required=False, default=None)
    parser.add_argument("--append", dest="append", default=False, action="store_true")
    parser.add_argument(
        "--recomputesdf", dest="recomputesdf", default=False, action="store_true"
    )
    parser.add_argument(
        "--skipduplicates", dest="skipduplicates", default=False, action="store_true"
    )

    args = parser.parse_args()

    input_paths = args.input_paths
    start_index = args.start
    end_index_inclusive = args.end
    output_path = args.dst
    append = args.append
    recompute_sdf = args.recomputesdf
    skip_duplicates = args.skipduplicates

    main(
        input_paths=input_paths,
        start_index=start_index,
        end_index_inclusive=end_index_inclusive,
        output_path=output_path,
        append=append,
        recompute_sdf=recompute_sdf,
        skip_duplicates=skip_duplicates,
    )
