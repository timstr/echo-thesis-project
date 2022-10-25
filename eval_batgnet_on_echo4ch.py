import fix_dead_command_line
import cleanup_when_killed

import numpy as np
import torch
import os
from argparse import ArgumentParser

from batgnet import BatGNet

from assert_eq import assert_eq
from network_utils import evaluate_prediction
from signals_and_geometry import backfill_occupancy
from torch_utils import restore_module
from which_device import get_compute_device
from device_dict import DeviceDict
from utils import progress_bar
from torch.utils.data._utils.collate import default_collate
from Echo4ChDatasetH5 import Echo4ChDataset, k_spectrograms, k_occupancy
from dataset_adapters import occupancy_grid_to_depthmap


def uint8_to_float(x):
    assert isinstance(x, torch.Tensor)
    assert_eq(x.dtype, torch.uint8)
    return x.float() / 255


def bool_to_float(x):
    assert isinstance(x, torch.Tensor)
    assert_eq(x.dtype, torch.bool)
    return x.float()


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        dest="experiment",
        required=True,
        help="short description or mnemonic of reason for training, used in log files and model names",
    )
    parser.add_argument(
        "--restoremodelpath", type=str, dest="restoremodelpath", required=True
    )

    args = parser.parse_args()

    k_env_dataset_test = "ECHO4CH_DATASET_TEST"

    dataset_test_path = os.environ.get(k_env_dataset_test)
    if dataset_test_path is None or not os.path.isfile(dataset_test_path):
        raise Exception(
            f"Please set the {k_env_dataset_test} environment variable to point to the ECHO4CH dataset HDF5 file for testing"
        )

    dataset_test = Echo4ChDataset(path_to_h5file=dataset_test_path)

    model = BatGNet()

    restore_module(model, args.restoremodelpath)
    model = model.to(get_compute_device())
    model.eval()

    with torch.no_grad():
        total_metrics = {}
        N = len(dataset_test)
        for i, dd in enumerate(dataset_test):
            spectrograms = uint8_to_float(dd[k_spectrograms]).to(get_compute_device())

            occupancy_gt = backfill_occupancy(dd[k_occupancy]).to(get_compute_device())
            assert_eq(occupancy_gt.shape, (64, 64, 64))

            occupancy_pred = model(spectrograms.unsqueeze(0)).squeeze(0)
            assert_eq(occupancy_pred.shape, (64, 64, 64))

            occupancy_pred = occupancy_pred >= 0.5

            metrics = evaluate_prediction(
                occupancy_gt=occupancy_gt, occupancy_pred=occupancy_pred
            )
            assert isinstance(metrics, dict)
            for k, v in metrics.items():
                assert isinstance(v, float)
                if not k in total_metrics:
                    total_metrics[k] = []
                total_metrics[k].append(v)

            progress_bar(i, N)

            # HACK
            if i == 100:
                break

        mean_metrics = {}
        for k, v in total_metrics.items():
            mean_metrics[k] = np.mean(v)

    for k, v in mean_metrics.items():
        print(f"{k}:\n    {v}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # I don't wanna see any stack traces
        pass
