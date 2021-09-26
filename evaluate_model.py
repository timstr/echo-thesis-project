import fix_dead_command_line
import cleanup_when_killed

import math
import os
from argparse import ArgumentParser
import numpy as np

import torch

from time_of_flight_net import TimeOfFlightNet
from Batvision.Models import (
    WaveformNet as BatVisionWaveform,
    SpectrogramNet as BatVisionSpectrogram,
)
from batgnet import BatGNet

from network_utils import (
    evaluate_prediction,
    evaluate_tofnet_on_whole_dataset,
    split_network_prediction,
)
from split_till_it_fits import SplitSize, split_till_it_fits
from assert_eq import assert_eq
from signals_and_geometry import (
    backfill_depthmap,
    backfill_occupancy,
    make_fm_chirp,
    sdf_to_occupancy,
)
from utils import progress_bar
from current_simulation_description import (
    all_grid_locations,
    make_receiver_indices,
    make_simulation_description,
    minimum_x_units,
)
from torch_utils import restore_module
from which_device import get_compute_device
from dataset3d import WaveDataset3d, k_sdf, k_sensor_recordings
from dataset_adapters import (
    convolve_recordings_dict,
    # sclog_dict,
    subset_recordings_dict,
    wavesim_to_batgnet_occupancy,
    wavesim_to_batgnet_spectrogram,
    wavesim_to_batvision_depthmap,
    wavesim_to_batvision_spectrogram,
    wavesim_to_batvision_waveform,
)

model_tof_net = "tofnet"
model_batvision_waveform = "batvision_waveform"
model_batvision_spectrogram = "batvision_spectrogram"
model_batgnet = "batgnet"


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        dest="experiment",
        required=True,
        help="short description or mnemonic of model, used in output file names",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            model_tof_net,
            model_batvision_waveform,
            model_batvision_spectrogram,
            model_batgnet,
        ],
        dest="model",
        required=True,
        default=model_tof_net,
    )
    parser.add_argument(
        "--tofcropsize",
        type=int,
        dest="tofcropsize",
        default=256,
        help="Number of samples used in time-of-flight crop",
    )

    parser.add_argument(
        "--chirpf0",
        type=float,
        dest="chirpf0",
        default=18_000.0,
        help="chirp start frequency (Hz)",
    )
    parser.add_argument(
        "--chirpf1",
        type=float,
        dest="chirpf1",
        default=22_000.0,
        help="chirp end frequency (Hz)",
    )
    parser.add_argument(
        "--chirplen",
        type=float,
        dest="chirplen",
        default=0.001,
        help="chirp duration (seconds)",
    )
    parser.add_argument("--receivercountx", type=int, dest="receivercountx", default=2)
    parser.add_argument("--receivercounty", type=int, dest="receivercounty", default=2)
    parser.add_argument("--receivercountz", type=int, dest="receivercountz", default=2)
    parser.add_argument(
        "--restoremodelpath", type=str, dest="restoremodelpath", required=True
    )
    parser.add_argument(
        "--backfill",
        dest="backfill",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--offsetsdf",
        dest="offsetsdf",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    description = make_simulation_description()

    model_type = args.model

    if model_type != model_tof_net:
        assert_eq(args.receivercountx, 1)
        assert_eq(args.receivercounty, 2)
        assert_eq(args.receivercountz, 2)

    sensor_indices = make_receiver_indices(
        args.receivercountx,
        args.receivercounty,
        args.receivercountz,
    )

    k_env_dataset_test = "WAVESIM_DATASET_TEST"

    dataset_train_path = os.environ.get(k_env_dataset_test)
    if dataset_train_path is None or not os.path.isfile(dataset_train_path):
        raise Exception(
            f"Please set the {k_env_dataset_test} environment variable to point to the WaveSim dataset HDF5 file for testing"
        )

    k_env_dataset_val = "WAVESIM_DATASET_VALIDATION"

    dataset_val_path = os.environ.get(k_env_dataset_val)
    if dataset_val_path is None or not os.path.isfile(dataset_val_path):
        raise Exception(
            f"Please set the {k_env_dataset_val} environment variable to point to the WaveSim dataset HDF5 file for validation"
        )

    fm_chirp = make_fm_chirp(
        begin_frequency_Hz=args.chirpf0,
        end_frequency_Hz=args.chirpf1,
        sampling_frequency=description.output_sampling_frequency,
        chirp_length_samples=math.ceil(
            args.chirplen * description.output_sampling_frequency
        ),
        wave="sine",
        device=get_compute_device(),
    )

    def adapt_signals(dd):
        dd = convolve_recordings_dict(
            subset_recordings_dict(dd, sensor_indices), fm_chirp
        )
        # if model_type in [model_tof_net, model_batvision_waveform]:
        #     dd = sclog_dict(dd)
        return dd

    if model_type == model_tof_net:
        model = TimeOfFlightNet(
            speed_of_sound=description.air_properties.speed_of_sound,
            sampling_frequency=description.output_sampling_frequency,
            recording_length_samples=description.output_length,
            crop_length_samples=args.tofcropsize,
            emitter_location=description.emitter_location,
            receiver_locations=description.sensor_locations[sensor_indices],
        )
    elif model_type == model_batvision_waveform:
        model = BatVisionWaveform(generator="direct")
    elif model_type == model_batvision_spectrogram:
        model = BatVisionSpectrogram(generator="unet")
    elif model_type == model_batgnet:
        model = BatGNet()

    restore_module(model, args.restoremodelpath)
    model = model.to(get_compute_device())
    model.eval()

    if model_type == model_tof_net and args.offsetsdf:
        with WaveDataset3d(description, dataset_val_path) as dataset_val:
            split_size_val = SplitSize("dense network evaluation for validation")
            possible_offsets = np.linspace(0.00, 0.05, num=51)
            best_offset = None
            best_metric = None
            for offset in possible_offsets:
                metric = evaluate_tofnet_on_whole_dataset(
                    the_model=model,
                    dataset=dataset_val,
                    description=description,
                    validationdownsampling=4,
                    adapt_signals_fn=adapt_signals,
                    sdf_offset=offset,
                    split_size=split_size_val,
                    backfill=args.backfill,
                )["f1score"]

                if best_metric is None or (metric > best_metric):
                    best_metric = metric
                    best_offset = offset

            assert best_offset is not None
            print(
                f"Using an sdf offset of {best_offset} which performed best on the validation set"
            )
    else:
        best_offset = 0.0

    split_size = SplitSize("dense network evaluation")

    locations = all_grid_locations(
        get_compute_device(), description, downsample_factor=1
    )

    with WaveDataset3d(
        description=description, path_to_h5file=dataset_train_path
    ) as dataset_test:
        total_metrics = {}
        N = len(dataset_test)
        for i, dd in enumerate(dataset_test):
            example = adapt_signals(dd.to(get_compute_device()))

            if model_type == model_tof_net:
                sdf_gt = example[k_sdf][minimum_x_units:]
                occupancy_gt = sdf_to_occupancy(sdf_gt)

                # - densely evaluate at dataset resolution
                sdf_pred = split_till_it_fits(
                    split_network_prediction,
                    split_size,
                    model=model,
                    locations=locations,
                    recordings=example[k_sensor_recordings],
                    description=description,
                ).reshape(
                    description.Nx - minimum_x_units, description.Ny, description.Nz
                )
                occupancy_pred = sdf_to_occupancy(sdf_pred, threshold=best_offset)

                if args.backfill:
                    occupancy_gt = backfill_occupancy(occupancy_gt)
                    occupancy_pred = backfill_occupancy(occupancy_pred)

                current_metrics = evaluate_prediction(
                    occupancy_pred=occupancy_pred,
                    occupancy_gt=occupancy_gt,
                )
            elif model_type in [model_batvision_waveform, model_batvision_spectrogram]:
                assert args.backfill
                depthmap_gt = wavesim_to_batvision_depthmap(example)
                if model_type == model_batvision_waveform:
                    inputs = wavesim_to_batvision_waveform(example)
                else:
                    inputs = wavesim_to_batvision_spectrogram(example)
                depthmap_pred = model(inputs.unsqueeze(0)).squeeze(0)

                occupancy_gt = backfill_depthmap(depthmap_gt, Nx=128)
                occupancy_pred = backfill_depthmap(depthmap_pred, Nx=128)

                current_metrics = evaluate_prediction(
                    occupancy_pred=occupancy_pred, occupancy_gt=occupancy_gt
                )
            elif model_type == model_batgnet:
                occupancy_gt = wavesim_to_batgnet_occupancy(
                    example, backfill=args.backfill
                )
                inputs = wavesim_to_batgnet_spectrogram(example)
                occupancy_pred_float = model(inputs.unsqueeze(0)).squeeze(0)
                occupancy_pred = occupancy_pred_float >= 0.5

                current_metrics = evaluate_prediction(
                    occupancy_pred=occupancy_pred, occupancy_gt=occupancy_gt
                )

            for k, v in current_metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = [v]
                else:
                    total_metrics[k].append(v)

            progress_bar(i, N)

        average_metrics = {}
        for k, v in total_metrics.items():
            average_metrics[k] = np.mean(total_metrics[k])

        for k, v in average_metrics.items():
            print(f"{k}:\n    {v}\n")


if __name__ == "__main__":
    try:
        with torch.no_grad():
            main()
    except KeyboardInterrupt:
        # I don't wanna see any stack traces
        pass
