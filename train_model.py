import fix_dead_command_line
import cleanup_when_killed

import random
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import datetime
from argparse import ArgumentParser

from Batvision.Models import (
    WaveformNet as BatVisionWaveform,
    SpectrogramNet as BatVisionSpectrogram,
)

from batgnet import BatGNet

from assert_eq import assert_eq
from torch_utils import restore_module, save_module
from which_device import get_compute_device
from device_dict import DeviceDict
from dataset3d import WaveDataset3d, k_sensor_recordings, k_sdf
from dataset_adapters import (
    convolve_recordings_dict,
    occupancy_grid_to_depthmap,
    # sclog_dict,
    subset_recordings_dict,
    wavesim_to_batgnet_occupancy,
    wavesim_to_batgnet_spectrogram,
    wavesim_to_batvision_depthmap,
    wavesim_to_batvision_spectrogram,
    wavesim_to_batvision_waveform,
)
from time_of_flight_net import TimeOfFlightNet
from network_utils import (
    evaluate_batgnet_on_whole_dataset,
    evaluate_batvision_on_whole_dataset,
    evaluate_tofnet_on_whole_dataset,
)
from utils import progress_bar
from current_simulation_description import (
    make_random_training_locations,
    make_receiver_indices,
    make_simulation_description,
)
from torch.utils.data._utils.collate import default_collate
from signals_and_geometry import make_fm_chirp, sample_obstacle_map
from split_till_it_fits import SplitSize, split_till_it_fits
from visualization import (
    colourize_sdf,
    render_slices_ground_truth,
    render_slices_prediction,
)


def concat_images(img1, img2, *rest, horizontal=True):
    imgs = [img1, img2]
    imgs.extend(rest)
    imgs = [
        torch.clamp(
            img.unsqueeze(0).repeat(3, 1, 1) if (img.ndim == 2) else img,
            min=0.0,
            max=1.0,
        )
        for img in imgs
    ]
    return torchvision.utils.make_grid(imgs, nrow=(len(imgs) if horizontal else 1))


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
        help="short description or mnemonic of reason for training, used in log files and model names",
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
        required=False,
        default=model_tof_net,
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
        default=1e-4,
        help="Adam optimizer learning rate",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        dest="adam_beta1",
        default=0.9,
        help="Adam optimizer parameter beta 1",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        dest="adam_beta2",
        default=0.999,
        help="Adam optimizer parameter beta 2",
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
        default=0.001,
        help="chirp duration (seconds)",
    )

    parser.add_argument(
        "--use_convolutions",
        dest="use_convolutions",
        default=False,
        action="store_true",
        help="(tofnet only) if specified, network will use convolutional layers. Otherwise fully-connected layers are used",
    )
    parser.add_argument(
        "--use_fourier_transform",
        dest="use_fourier_transform",
        default=False,
        action="store_true",
        help="(tofnet only) if specified, a fourier transform is applied to the input audio before the network receives it",
    )
    parser.add_argument(
        "--hidden_features",
        type=int,
        dest="hidden_features",
        default=128,
        help="(tofnet only) size of the hidden feature dimension of the neural network, convolutional or fully-connected",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        dest="kernel_size",
        default=128,
        help="(tofnet only) size of the convolutional kernels of the neural network, if convolutions are being used",
    )

    parser.add_argument(
        "--samplesperexample",
        type=int,
        dest="samplesperexample",
        default=128,
        help="number of spatial locations per example to train on, similar to batch size, but per-example, not per-batch",
    )
    parser.add_argument(
        "--backfill",
        dest="backfill",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--nosave",
        dest="nosave",
        default=False,
        action="store_true",
        help="do not save model files",
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
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_device,
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

    validation_splits = SplitSize("compute_validation_metrics")

    def compute_validation_metrics(the_model):
        validation_begin_time = datetime.datetime.now()
        print("Computing validation metrics...")

        if model_type == model_tof_net:
            metrics = evaluate_tofnet_on_whole_dataset(
                the_model=the_model,
                dataset=dataset_val,
                description=description,
                validationdownsampling=args.validationdownsampling,
                adapt_signals_fn=adapt_signals,
                sdf_offset=0.0,
                backfill=args.backfill,
                split_size=validation_splits,
            )
            primary_metric_name = "mean_absolute_error_sdf"
        elif model_type in [model_batvision_waveform, model_batvision_spectrogram]:
            if model_type == model_batvision_waveform:
                batvision_mode = "waveform"
            else:
                batvision_mode = "spectrogram"
            mae = evaluate_batvision_on_whole_dataset(
                the_model=the_model,
                dataset=dataset_val,
                description=description,
                adapt_signals_fn=adapt_signals,
                batvision_mode=batvision_mode,
            )
            primary_metric_name = "mean_absolute_error_depthmap"
            metrics = {primary_metric_name: mae}
        elif model_type == model_batgnet:
            mse = evaluate_batgnet_on_whole_dataset(
                the_model=the_model,
                dataset=dataset_val,
                description=description,
                adapt_signals_fn=adapt_signals,
                backfill=args.backfill,
            )
            primary_metric_name = "mean_squared_error_occupancy"
            metrics = {primary_metric_name: mse}
        else:
            raise Exception("Unrecognized model type")
        validation_end_time = datetime.datetime.now()
        duration = validation_end_time - validation_begin_time
        seconds = float(duration.seconds) + (duration.microseconds / 1_000_000.0)
        print(f"Computing validation done after {seconds} seconds.")
        return metrics, primary_metric_name

    def plot_images(the_model):
        visualization_begin_time = datetime.datetime.now()
        print("Generating visualizations...")

        example_train = adapt_signals(
            random.choice(dataset_train).to(get_compute_device())
        )
        example_val = adapt_signals(random.choice(dataset_val).to(get_compute_device()))

        if model_type == model_tof_net:
            vis_locations = make_random_training_locations(
                sdf_batch=example_train[k_sdf].unsqueeze(0),
                samples_per_example=args.samplesperexample,
                device=get_compute_device(),
                description=description,
            ).squeeze(0)

            # plot the ground truth obstacles
            slices_train_gt = render_slices_ground_truth(
                example_train[k_sdf],
                description,
                locations=vis_locations,
                colour_function=colourize_sdf,
            )

            progress_bar(0, 4)

            slices_train_pred = split_till_it_fits(
                render_slices_prediction,
                sdf_slice_prediction_splits,
                the_model,
                example_train[k_sensor_recordings],
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
                example_val[k_sdf],
                description,
                colour_function=colourize_sdf,
            )

            progress_bar(2, 4)

            slices_val_pred = split_till_it_fits(
                render_slices_prediction,
                sdf_slice_prediction_splits,
                the_model,
                example_val[k_sensor_recordings],
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

        elif model_type in [model_batvision_waveform, model_batvision_spectrogram]:
            train_depthmap_gt = wavesim_to_batvision_depthmap(example_train)
            val_depthmap_gt = wavesim_to_batvision_depthmap(example_val)

            if model_type == model_batvision_waveform:
                make_input = wavesim_to_batvision_waveform
            else:
                make_input = wavesim_to_batvision_spectrogram

            train_depthmap_pred = the_model(
                make_input(example_train).unsqueeze(0)
            ).squeeze(0)
            val_depthmap_pred = the_model(make_input(example_val).unsqueeze(0)).squeeze(
                0
            )

            train_depthmaps = concat_images(train_depthmap_gt, train_depthmap_pred)
            val_depthmaps = concat_images(val_depthmap_gt, val_depthmap_pred)

            writer.add_image(
                "Depthmap Ground Truth, Depthmap Prediction (train)",
                train_depthmaps,
                global_iteration,
            )
            writer.add_image(
                "Depthmap Ground Truth, Depthmap Prediction (validation)",
                val_depthmaps,
                global_iteration,
            )

        elif model_type == model_batgnet:

            threshold = 0.5

            def make_image(the_example):
                occupancy_gt = wavesim_to_batgnet_occupancy(
                    the_example, backfill=args.backfill
                )
                assert_eq(occupancy_gt.shape, (64, 64, 64))
                occupancy_pred = the_model(
                    wavesim_to_batgnet_spectrogram(the_example).unsqueeze(0)
                ).squeeze(0)
                assert_eq(occupancy_pred.shape, (64, 64, 64))
                occupancy_pred_binary = occupancy_pred >= threshold
                depthmaps_gt = concat_images(
                    occupancy_grid_to_depthmap(occupancy_gt.flip(0), 0).permute(1, 0),
                    occupancy_grid_to_depthmap(occupancy_gt, 1).permute(1, 0),
                    occupancy_grid_to_depthmap(occupancy_gt, 2).permute(1, 0),
                    horizontal=False,
                )
                depthmaps_pred = concat_images(
                    occupancy_grid_to_depthmap(
                        occupancy_pred_binary.flip(0), 0
                    ).permute(1, 0),
                    occupancy_grid_to_depthmap(occupancy_pred_binary, 1).permute(1, 0),
                    occupancy_grid_to_depthmap(occupancy_pred_binary, 2).permute(1, 0),
                    horizontal=False,
                )
                projections_pred = concat_images(
                    torch.mean(occupancy_pred, dim=0).permute(1, 0),
                    torch.mean(occupancy_pred, dim=1).permute(1, 0),
                    torch.mean(occupancy_pred, dim=2).permute(1, 0),
                    horizontal=False,
                )
                slices_pred = concat_images(
                    occupancy_pred[occupancy_pred.shape[0] // 2, :, :].permute(1, 0),
                    occupancy_pred[:, occupancy_pred.shape[1] // 2, :].permute(1, 0),
                    occupancy_pred[:, :, occupancy_pred.shape[2] // 2].permute(1, 0),
                    horizontal=False,
                )

                return concat_images(
                    depthmaps_gt, depthmaps_pred, projections_pred, slices_pred
                )

            writer.add_image(
                "Depthmaps Ground Truth, Depthmap Prediction, Projections Prediction, Slices Prediction (train)",
                make_image(example_train),
                global_iteration,
            )
            writer.add_image(
                "Depthmaps Ground Truth, Depthmap Prediction, Projections Prediction, Slices Prediction (validation)",
                make_image(example_val),
                global_iteration,
            )

        visualization_end_time = datetime.datetime.now()
        duration = visualization_end_time - visualization_begin_time
        seconds = float(duration.seconds) + (duration.microseconds / 1_000_000.0)
        print(f"Generating visualizations done after {seconds} seconds.")

    if model_type == model_tof_net:
        model = TimeOfFlightNet(
            speed_of_sound=description.air_properties.speed_of_sound,
            sampling_frequency=description.output_sampling_frequency,
            recording_length_samples=description.output_length,
            crop_length_samples=args.tofcropsize,
            emitter_location=description.emitter_location,
            receiver_locations=description.sensor_locations[sensor_indices],
            hidden_features=args.hidden_features,
            kernel_size=args.kernel_size,
            use_convolutions=args.use_convolutions,
            use_fourier_transform=args.use_fourier_transform,
        )
    elif model_type == model_batvision_waveform:
        model = BatVisionWaveform(generator="direct")
    elif model_type == model_batvision_spectrogram:
        model = BatVisionSpectrogram(generator="unet")
    elif model_type == model_batgnet:
        model = BatGNet()

    model = model.to(get_compute_device())

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learningrate,
        betas=(args.adam_beta1, args.adam_beta2),
    )

    if args.restoremodelpath is not None:
        restore_module(model, args.restoremodelpath)
        restore_module(optimizer, args.restoreoptimizerpath)

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    if not args.nosave:
        base_model_path = os.environ.get("TRAINING_MODEL_PATH")

        if base_model_path is None:
            raise Exception(
                "Please set the TRAINING_MODEL_PATH environment variable to point to the desired model directory"
            )
        if not os.path.exists(base_model_path):
            os.makedirs(base_model_path)

        model_path = os.path.join(base_model_path, f"{args.experiment}_{timestamp}")
        if os.path.exists(model_path):
            raise Exception(
                f'Error: attempted to create a folder for saving models at "{model_path}" but the folder already exists.'
            )

        os.makedirs(model_path)

    log_path_root = os.environ.get("TRAINING_LOG_PATH")

    if log_path_root is None:
        raise Exception(
            "Please set the TRAINING_LOG_PATH environment variable to point to the desired log directory"
        )

    if not os.path.exists(log_path_root):
        os.makedirs(log_path_root)

    log_folder_name = f"{args.experiment}_{timestamp}"

    log_path = os.path.join(log_path_root, log_folder_name)

    def save_things(suffix=None):
        assert suffix is None or isinstance(suffix, str)
        if args.nosave:
            return
        save_module(
            model,
            os.path.join(
                model_path,
                f"model_{model_type}_{suffix or ''}.dat",
            ),
        )
        save_module(
            optimizer,
            os.path.join(
                model_path,
                f"optimizer_{model_type}{'' if suffix is None else ('_' + suffix)}.dat",
            ),
        )

    global_iteration = 0

    try:
        with SummaryWriter(log_path) as writer:

            sdf_slice_prediction_splits = SplitSize("SDF slice prediction")

            num_epochs = 1000000
            val_loss_y = []
            val_loss_x = []
            best_val_mse = np.inf

            for i_epoch in range(num_epochs):
                train_iter = iter(train_loader)
                for i_example in range(len(train_loader)):
                    example_batch = next(train_iter).to(get_compute_device())
                    example_batch = adapt_signals(example_batch)

                    if model_type == model_tof_net:
                        sdf = example_batch[k_sdf]

                        locations = make_random_training_locations(
                            sdf,
                            samples_per_example=args.samplesperexample,
                            device=get_compute_device(),
                            description=description,
                        )

                        gt = sample_obstacle_map(sdf, locations, description)
                        pred = model(example_batch[k_sensor_recordings], locations)
                        assert_eq(gt.shape, pred.shape)
                        loss = torch.mean(torch.abs(pred - gt))

                    elif model_type in [
                        model_batvision_waveform,
                        model_batvision_spectrogram,
                    ]:
                        if model_type == model_batvision_waveform:
                            inputs = wavesim_to_batvision_waveform(example_batch)
                        else:
                            inputs = wavesim_to_batvision_spectrogram(example_batch)
                        gt = wavesim_to_batvision_depthmap(example_batch)
                        pred = model(inputs)
                        assert_eq(gt.shape, pred.shape)
                        loss = torch.mean(torch.abs(pred - gt))
                    elif model_type == model_batgnet:
                        inputs = wavesim_to_batgnet_spectrogram(example_batch)
                        gt = wavesim_to_batgnet_occupancy(
                            example_batch, backfill=args.backfill
                        ).float()
                        pred = model(inputs)
                        assert_eq(gt.shape, pred.shape)
                        loss = torch.mean(torch.square(pred - gt))

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

                    loss = loss.item()

                    writer.add_scalar("training loss", loss, global_iteration)

                    progress_bar(
                        (global_iteration) % args.plotinterval, args.plotinterval
                    )

                    validation_time = (
                        ((global_iteration + 1) % args.validationinterval) == 0
                    ) or (global_iteration == 0)

                    if validation_time:
                        model.eval()
                        (
                            curr_val_metrics,
                            primary_metrics_name,
                        ) = compute_validation_metrics(model)
                        model.train()
                        curr_primary_val_metric = curr_val_metrics[primary_metrics_name]
                        if curr_primary_val_metric < best_val_mse:
                            best_val_mse = curr_primary_val_metric
                            save_things("best")

                        save_things("latest")

                        val_loss_x.append(global_iteration)
                        val_loss_y.append(curr_primary_val_metric)

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
        if args.nosave:
            print(
                "\n\nControl-C detected, but not saving model due to --nosave option\n"
            )
        else:
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
