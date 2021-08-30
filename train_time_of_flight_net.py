import fix_dead_command_line
import cleanup_when_killed

import math
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import os
import datetime
from argparse import ArgumentParser

from the_device import the_device
from device_dict import DeviceDict
from dataset3d import WaveDataset3d
from time_of_flight_net import TimeOfFlightNet
from tof_utils import (
    SplitSize,
    all_grid_locations,
    colourize_sdf,
    make_fm_chirp,
    convolve_recordings,
    evaluate_prediction,
    make_random_training_locations,
    make_receiver_indices,
    render_slices_ground_truth,
    render_slices_prediction,
    restore_module,
    sample_obstacle_map,
    save_module,
    split_network_prediction,
    split_till_it_fits,
)
from utils import progress_bar
from current_simulation_description import (
    make_simulation_description,
    minimum_x_units,
)
from plot_utils import LossPlotter, plt_screenshot
from torch.utils.data._utils.collate import default_collate


def concat_images(img1, img2):
    return torchvision.utils.make_grid([img1, img2], nrow=2)


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
        "--batchsize",
        type=int,
        dest="batchsize",
        default=4,
        help="batch size used for each training iteration",
    )
    parser.add_argument(
        "--learningrate",
        type=float,
        dest="learningrate",
        default=1e-3,
        help="optimizer learning rate",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        dest="iterations",
        default=None,
        help="if specified, number of iterations to train for. Trains forever if not specified.",
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
        default=0.002,
        help="chirp duration (seconds)",
    )

    parser.add_argument(
        "--samplesperexample",
        type=int,
        dest="samplesperexample",
        default=128,
        help="number of spatial locations per example to train on, similar to batch size, but per-example, not per-batch",
    )
    parser.add_argument(
        "--nosave",
        dest="nosave",
        default=False,
        action="store_true",
        help="do not save model files",
    )
    parser.add_argument(
        "--raymarch",
        dest="raymarch",
        default=False,
        action="store_true",
        help="when plotting, render raymarched 3D images",
    )
    parser.add_argument(
        "--plotinterval",
        type=int,
        dest="plotinterval",
        default=512,
        help="number of training iterations between generating visualizations",
    )
    parser.add_argument(
        "--validationinterval",
        type=int,
        dest="validationinterval",
        default=256,
        help="number of training iterations between computating validation metrics",
    )
    parser.add_argument(
        "--validationdownsampling",
        dest="validationdownsampling",
        type=int,
        default=1,
        help="factor by which to downsample space when densely computing validation metrics, relative to full dataset resolution",
    )
    parser.add_argument("--receivercountx", type=int, dest="receivercountx", default=2)
    parser.add_argument("--receivercounty", type=int, dest="receivercounty", default=2)
    parser.add_argument("--receivercountz", type=int, dest="receivercountz", default=2)
    parser.add_argument(
        "--restoremodelpath", type=str, dest="restoremodelpath", default=None
    )
    parser.add_argument(
        "--restoreoptimizerpath", type=str, dest="restoreoptimizerpath", default=None
    )

    args = parser.parse_args()
    description = make_simulation_description()

    assert (args.restoremodelpath is None) == (args.restoreoptimizerpath is None)

    sensor_indices = make_receiver_indices(
        args.receivercountx,
        args.receivercounty,
        args.receivercountz,
    )

    print(f"Using {len(sensor_indices)} receivers in total")

    k_env_dataset_train = "WAVESIM_DATASET_TRAIN"
    k_env_dataset_val = "WAVESIM_DATASET_VALIDATION"

    dataset_train_path = os.environ.get(k_env_dataset_train)
    if dataset_train_path is None or not os.path.isfile(dataset_train_path):
        raise Exception(
            f"Please set the {k_env_dataset_train} environment variable to point to the WaveSim dataset HDF5 file for training"
        )

    dataset_val_path = os.environ.get(k_env_dataset_val)
    if dataset_val_path is None or not os.path.isfile(dataset_val_path):
        raise Exception(
            f"Please set the {k_env_dataset_val} environment variable to point to the WaveSim dataset HDF5 file for validation"
        )

    dataset_train = WaveDataset3d(
        description=description, path_to_h5file=dataset_train_path
    )
    dataset_val = WaveDataset3d(
        description=description, path_to_h5file=dataset_val_path
    )

    def collate_fn_device(batch):
        return DeviceDict(default_collate(batch))

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batchsize,
        num_workers=0,
        pin_memory=False,  # Note, setting pin_memory=False to avoid the pin_memory call
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_device,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_device,
    )

    fm_chirp = (
        torch.tensor(
            make_fm_chirp(
                begin_frequency_Hz=args.chirpf0,
                end_frequency_Hz=args.chirpf1,
                sampling_frequency=description.output_sampling_frequency,
                chirp_length_samples=math.ceil(
                    args.chirplen * description.output_sampling_frequency
                ),
                wave="sine",
            )
        )
        .float()
        .to(the_device)
    )

    validation_splits = SplitSize("compute_validation_metrics")

    def compute_validation_metrics(the_model):
        assert isinstance(the_model, nn.Module)
        with torch.no_grad():
            validation_begin_time = datetime.datetime.now()
            print("Computing validation metrics...")
            total_metrics = {}
            locations = all_grid_locations(
                the_device, description, downsample_factor=args.validationdownsampling
            )
            x_steps = (description.Nx - minimum_x_units) // args.validationdownsampling
            y_steps = description.Ny // args.validationdownsampling
            z_steps = description.Nz // args.validationdownsampling
            N = len(dataset_val)
            for i, dd in enumerate(dataset_val):
                sdf_gt = sample_obstacle_map(
                    obstacle_map_batch=dd["sdf"].unsqueeze(0).to(the_device),
                    locations_xyz_batch=locations.unsqueeze(0),
                    description=description,
                )
                sdf_gt = sdf_gt.reshape(x_steps, y_steps, z_steps)
                recordings_ir = dd["sensor_recordings"][sensor_indices].to(the_device)

                recordings_fm = convolve_recordings(
                    fm_chirp, recordings_ir, description
                )

                sdf_pred = split_till_it_fits(
                    split_network_prediction,
                    validation_splits,
                    model=the_model,
                    locations=locations,
                    recordings=recordings_fm,
                    description=description,
                )
                sdf_pred = sdf_pred.reshape(x_steps, y_steps, z_steps)
                metrics = evaluate_prediction(
                    sdf_pred=sdf_pred,
                    sdf_gt=sdf_gt,
                    description=description,
                    downsample_factor=args.validationdownsampling,
                )

                assert isinstance(metrics, dict)
                for k, v in metrics.items():
                    assert isinstance(v, float)
                    if not k in total_metrics:
                        total_metrics[k] = []
                    total_metrics[k].append(v)
                progress_bar(i, N)

            mean_metrics = {}
            for k, v in total_metrics.items():
                mean_metrics[k] = np.mean(v)

            validation_end_time = datetime.datetime.now()
            duration = validation_end_time - validation_begin_time
            seconds = float(duration.seconds) + (duration.microseconds / 1_000_000.0)
            print(f"Computing validation done after {seconds} seconds.")

            return mean_metrics

    def plot_images(the_model):
        visualization_begin_time = datetime.datetime.now()
        print("Generating visualizations...")

        val_batch_cpu = next(iter(val_loader))
        val_batch_gpu = val_batch_cpu.to(the_device)

        # plot the ground truth obstacles
        slices_train_gt = render_slices_ground_truth(
            batch_gpu["sdf"][0],
            description,
            locations=locations[0].to(the_device),
            colour_function=colourize_sdf,
        )

        progress_bar(0, 4)

        plot_recordings_train = convolve_recordings(
            fm_chirp=fm_chirp,
            sensor_recordings=batch_gpu["sensor_recordings"][0, sensor_indices],
            description=description,
        )

        slices_train_pred = split_till_it_fits(
            render_slices_prediction,
            sdf_slice_prediction_splits,
            the_model,
            plot_recordings_train,
            description,
            colour_function=colourize_sdf,
        )

        progress_bar(1, 4)

        slices_train = concat_images(slices_train_gt, slices_train_pred)

        writer.add_image(
            "SDF Ground Truth, SDF Prediction (train)",
            slices_train,
            global_iteration,
        )

        slices_val_gt = render_slices_ground_truth(
            val_batch_gpu["sdf"][0],
            description,
            colour_function=colourize_sdf,
        )

        progress_bar(2, 4)

        plot_recordings_val = convolve_recordings(
            fm_chirp=fm_chirp,
            sensor_recordings=val_batch_gpu["sensor_recordings"][0, sensor_indices],
            description=description,
        )
        slices_val_pred = split_till_it_fits(
            render_slices_prediction,
            sdf_slice_prediction_splits,
            the_model,
            plot_recordings_val,
            description,
            colour_function=colourize_sdf,
        )

        progress_bar(3, 4)

        slices_val = concat_images(slices_val_gt, slices_val_pred)
        writer.add_image(
            "SDF Ground Truth, SDF Prediction (validation)",
            slices_val,
            global_iteration,
        )

        visualization_end_time = datetime.datetime.now()
        duration = visualization_end_time - visualization_begin_time
        seconds = float(duration.seconds) + (duration.microseconds / 1_000_000.0)
        print(f"Generating visualizations done after {seconds} seconds.")

    model = TimeOfFlightNet(
        speed_of_sound=description.air_properties.speed_of_sound,
        sampling_frequency=description.output_sampling_frequency,
        recording_length_samples=description.output_length,
        crop_length_samples=args.tofcropsize,
        emitter_location=description.emitter_location,
        receiver_locations=description.sensor_locations[sensor_indices],
    ).to(the_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningrate)

    if args.restoremodelpath is not None:
        restore_module(model, args.restoremodelpath)
        restore_module(optimizer, args.restoreoptimizerpath)

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    base_model_path = os.environ.get("TRAINING_MODEL_PATH")

    if base_model_path is None or not os.path.exists(base_model_path):
        raise Exception(
            "Please set the TRAINING_MODEL_PATH environment variable to point to the desired model directory"
        )

    model_path = os.path.join(base_model_path, f"{args.experiment}_{timestamp}")
    if os.path.exists(model_path):
        raise Exception(
            f'Error: attempted to create a folder for saving models at "{model_path}" but the folder already exists.'
        )

    os.makedirs(model_path)

    log_path_root = os.environ.get("TRAINING_LOG_PATH")

    log_folder_name = f"{args.experiment}_{timestamp}"

    log_path = os.path.join(log_path_root, log_folder_name)

    def save_things(suffix=None):
        assert suffix is None or isinstance(suffix, str)
        save_module(
            model,
            os.path.join(
                model_path,
                f"model_{suffix or ''}.dat",
            ),
        )
        save_module(
            optimizer,
            os.path.join(
                model_path,
                f"optimizer{'' if suffix is None else ('_' + suffix)}.dat",
            ),
        )

    global_iteration = 0

    try:
        with SummaryWriter(log_path) as writer:

            sdf_slice_prediction_splits = SplitSize("SDF slice prediction")

            num_epochs = 1000000
            train_loss_plotter = LossPlotter(
                aggregation_interval=256, colour=(0.0, 0.0, 1.0), num_quantiles=10
            )
            val_loss_y = []
            val_loss_x = []
            best_val_mse = np.inf

            for i_epoch in range(num_epochs):
                train_iter = iter(train_loader)
                for i_example in range(len(train_loader)):
                    batch_cpu = next(train_iter)
                    batch_gpu = batch_cpu.to(the_device)

                    sensor_recordings_ir = batch_gpu["sensor_recordings"][
                        :, sensor_indices
                    ]

                    sensor_recordings_fm = convolve_recordings(
                        fm_chirp, sensor_recordings_ir, description
                    )

                    sdf = batch_gpu["sdf"]

                    locations = make_random_training_locations(
                        sdf,
                        samples_per_example=args.samplesperexample,
                        device=the_device,
                        description=description,
                    )

                    ground_truth = sample_obstacle_map(sdf, locations, description)

                    pred_gpu = model(sensor_recordings_fm, locations)

                    loss = torch.nn.functional.mse_loss(ground_truth, pred_gpu)

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

                    loss = loss.item()

                    train_loss_plotter.append(loss)

                    writer.add_scalar("training loss", loss, global_iteration)

                    progress_bar(
                        (global_iteration) % args.plotinterval, args.plotinterval
                    )

                    validation_time = (
                        ((global_iteration + 1) % args.validationinterval) == 0
                    ) or (global_iteration == 0)

                    if validation_time:
                        model.eval()
                        curr_val_metrics = compute_validation_metrics(model)
                        model.train()
                        curr_val_mse = curr_val_metrics["mean_squared_error_sdf"]
                        if curr_val_mse < best_val_mse:
                            best_val_mse = curr_val_mse
                            if not args.nosave:
                                save_things("best")

                        if not args.nosave:
                            save_things("latest")

                        val_loss_x.append(global_iteration)
                        val_loss_y.append(curr_val_mse)

                        for k, v in curr_val_metrics.items():
                            writer.add_scalar(f"validation_{k}", v, global_iteration)

                    time_to_plot = (
                        (global_iteration + 1) % args.plotinterval
                    ) == 0 or (global_iteration == 0)

                    if time_to_plot:
                        model.eval()

                        print(
                            f"Epoch {i_epoch}, {global_iteration + 1} total iterations"
                        )

                        plot_images(model)

                        if (
                            args.iterations is not None
                            and global_iteration > args.iterations
                        ):
                            print("Done - desired iteration count was reached")
                            return

                        model.train()

                    global_iteration += 1

    except KeyboardInterrupt:
        print("\n\nControl-C detected, saving model...\n")
        save_things("aborted")
        print("Exiting")
        exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # I don't wanna see any stack traces
        pass
