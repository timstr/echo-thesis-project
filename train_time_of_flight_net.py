import fix_dead_command_line

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
    sample_obstacle_map,
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


def save_module(the_module, filename):
    print(f'Saving module to "{filename}"')
    torch.save(the_module.state_dict(), filename)


def restore_module(the_module, filename):
    print('Restoring module from "{}"'.format(filename))
    the_module.load_state_dict(torch.load(filename))


def main():
    parser = ArgumentParser()

    parser.add_argument("--experiment", type=str, dest="experiment", required=True)
    parser.add_argument("--batchsize", type=int, dest="batchsize", default=4)
    parser.add_argument("--learningrate", type=float, dest="learningrate", default=1e-3)
    parser.add_argument("--iterations", type=int, dest="iterations", default=None)
    parser.add_argument("--tofcropsize", type=int, dest="tofcropsize", default=128)
    parser.add_argument(
        "--samplesperexample", type=int, dest="samplesperexample", default=128
    )
    parser.add_argument("--nosave", dest="nosave", default=False, action="store_true")
    parser.add_argument(
        "--raymarch", dest="raymarch", default=False, action="store_true"
    )
    parser.add_argument("--plotinterval", type=int, dest="plotinterval", default=512)
    parser.add_argument(
        "--validationinterval", type=int, dest="validationinterval", default=256
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
        batch_size=1,  # NOTE: batch size is 1 because this is only used for plotting
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_device,
    )

    # NOTE: "maximum supported frequency: 17.15kHz" according to
    # the matlab version of k-wave for a similar setup
    fm_chirp = (
        torch.tensor(
            make_fm_chirp(
                begin_frequency_Hz=1_000.0,
                end_frequency_Hz=10_000.0,
                sampling_frequency=description.output_sampling_frequency,
                chirp_length_samples=math.ceil(
                    0.001 * description.output_sampling_frequency
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
            locations = all_grid_locations(the_device, description)
            N = len(dataset_val)
            for i, dd in enumerate(dataset_val):
                sdf_gt = dd["sdf"][minimum_x_units:].to(the_device)
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
                sdf_pred = sdf_pred.reshape(
                    (description.Nx - minimum_x_units), description.Ny, description.Nz
                )
                metrics = evaluate_prediction(
                    sdf_pred,
                    sdf_gt,
                    description,
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
            print(f"Computing validation metrics done after {seconds} seconds.")

            return mean_metrics

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

    def save_things(iteration, suffix=None):
        assert isinstance(iteration, int)
        assert suffix is None or isinstance(suffix, str)
        save_module(
            model,
            os.path.join(
                model_path,
                f"model_{iteration + 1}{suffix or ''}.dat",
            ),
        )
        save_module(
            optimizer,
            os.path.join(
                model_path,
                f"optimizer_{iteration + 1}{suffix or ''}.dat",
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
                            best_val_loss = curr_val_mse
                            if not args.nosave:
                                save_things(global_iteration, "_best")

                        # if not args.nosave:
                        # model.save(make_model_filename("latest"))

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

                        val_batch_cpu = next(iter(val_loader))
                        val_batch_gpu = val_batch_cpu.to(the_device)

                        # plot the ground truth obstacles
                        slices_train_gt = render_slices_ground_truth(
                            batch_gpu["sdf"][0],
                            description,
                            locations=locations[0].to(the_device),
                            colour_function=colourize_sdf,
                        )
                        slices_train_pred = split_till_it_fits(
                            render_slices_prediction,
                            sdf_slice_prediction_splits,
                            model,
                            batch_gpu["sensor_recordings"][0, sensor_indices],
                            description,
                            colour_function=colourize_sdf,
                        )
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
                        slices_val_pred = split_till_it_fits(
                            render_slices_prediction,
                            sdf_slice_prediction_splits,
                            model,
                            val_batch_gpu["sensor_recordings"][0, sensor_indices],
                            description,
                            colour_function=colourize_sdf,
                        )
                        slices_val = concat_images(slices_val_gt, slices_val_pred)
                        writer.add_image(
                            "SDF Ground Truth, SDF Prediction (validation)",
                            slices_val,
                            global_iteration,
                        )

                        if (
                            args.iterations is not None
                            and global_iteration > args.iterations
                        ):
                            print("Done - desired iteration count was reached")
                            return

                        model.train()

                    if ((global_iteration + 1) % 65536) == 0:
                        save_things(global_iteration, "_latest")

                    global_iteration += 1

    except KeyboardInterrupt:
        print("\n\nControl-C detected, saving model...\n")
        save_things(global_iteration, "_aborted")
        print("Exiting")
        exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # I don't wanna see any stack traces
        pass
