import os
import sys

from current_simulation_description import make_simulation_description
from dataset3d import WaveDataset3d


def main(dataset_path):
    assert isinstance(dataset_path, str)
    if not os.path.exists(dataset_path):
        print(f"The file {dataset_path} does not exist")
        return
    desc = make_simulation_description()
    with WaveDataset3d(desc, dataset_path) as ds:
        n = len(ds)
        print(f"The dataset contains {n} example{'' if n == 1 else 's'}.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/datset.h5")
        exit(-1)
    dataset_path = sys.argv[1]
    main(dataset_path)
