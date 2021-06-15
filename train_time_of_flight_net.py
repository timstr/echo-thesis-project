import fix_dead_command_line

import os
import datetime
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from the_device import the_device
from device_dict import DeviceDict
from dataset3d import WaveDataset3d
from time_of_flight_net import TimeOfFlightNet
from tof_utils import (
    make_random_locations,
    plot_ground_truth,
    plot_prediction,
    sample_obstacle_map,
)
from utils import progress_bar
from custom_collate_fn import custom_collate
from current_simulation_description import make_simulation_description
from plot_utils import LossPlotter, plt_screenshot


def main():
    parser = ArgumentParser()

    parser.add_argument("--experiment", type=str, dest="experiment", required=True)
    parser.add_argument("--batchsize", type=int, dest="batchsize", default=4)
    parser.add_argument("--iterations", type=int, dest="iterations", default=None)
    parser.add_argument("--tofcropsize", type=int, dest="tofcropsize", default=128)
    parser.add_argument(
        "--samplesperexample", type=int, dest="samplesperexample", default=128
    )
    parser.add_argument("--nosave", dest="nosave", default=False, action="store_true")
    parser.add_argument(
        "--nodisplay", dest="nodisplay", default=False, action="store_true"
    )
    parser.add_argument("--plotinterval", type=int, dest="plotinterval", default=32)
    parser.add_argument(
        "--validationinterval", type=int, dest="validationinterval", default=256
    )

    args = parser.parse_args()

    description = make_simulation_description()

    # for i in range(description.sensor_count):
    #     print(f"{i}    {description.sensor_locations[i]}")

    sensor_indices = range(16, 32, 1)

    dataset = WaveDataset3d(description, "dataset_v2.h5")

    val_ratio = 0.1
    val_size = int(len(dataset) * val_ratio)
    indices_val = list(range(0, val_size))
    indices_train = list(range(val_size, len(dataset)))

    val_set = torch.utils.data.Subset(dataset, indices_val)
    train_set = torch.utils.data.Subset(dataset, indices_train)

    def collate_fn_device(batch):
        return DeviceDict(custom_collate(batch))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batchsize,
        num_workers=0,
        pin_memory=False,  # Note, setting pin_memory=False to avoid the pin_memory call
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_device,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batchsize,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_device,
    )

    model = TimeOfFlightNet(
        speed_of_sound=description.air_properties.speed_of_sound,
        sampling_frequency=description.output_sampling_frequency,
        recording_length_samples=description.output_length,
        crop_length_samples=args.tofcropsize,
        emitter_location=description.emitter_location,
        receiver_locations=description.sensor_locations[sensor_indices],
    ).to(the_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model_path = os.environ.get("TRAINING_MODEL_PATH")

    if model_path is None or not os.path.exists(model_path):
        raise Exception(
            "Please set the TRAINING_MODEL_PATH environment variable to point to the desired model directory"
        )

    def make_model_filename(label):
        assert isinstance(label, str)
        fname = f"{args.experiment}_{timestamp}_{label}.dat"
        return os.path.join(model_path, fname)

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    log_path_root = os.environ.get("TRAINING_LOG_PATH")

    log_folder_name = f"{args.experiment}_{timestamp}"

    log_path = os.path.join(log_path_root, log_folder_name)

    with SummaryWriter(log_path) as writer:

        if not args.nodisplay:
            plt.ion()

        fig, axes = plt.subplots(2, 4, figsize=(22, 10), dpi=80)
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))

        ax_t1 = axes[0, 0]
        ax_t2 = axes[0, 1]
        ax_t3 = axes[0, 2]
        ax_t4 = axes[0, 3]
        ax_b1 = axes[1, 0]
        ax_b2 = axes[1, 1]
        ax_b3 = axes[1, 2]
        ax_b4 = axes[1, 3]
        num_epochs = 1000000
        train_loss_plotter = LossPlotter(
            aggregation_interval=32, colour=(0.0, 0.0, 1.0), num_quantiles=10
        )
        val_loss_y = []
        val_loss_x = []
        best_val_loss = np.inf
        global_iteration = 0
        for e in range(num_epochs):
            train_iter = iter(train_loader)
            for i in range(len(train_loader)):
                batch_cpu = next(train_iter)
                batch_gpu = batch_cpu.to(the_device)

                sensor_recordings = batch_gpu["sensor_recordings"][:, sensor_indices]

                locations = make_random_locations(
                    batch_size=args.batchsize,
                    samples_per_example=args.samplesperexample,
                    device=the_device,
                    description=description,
                )

                ground_truth = sample_obstacle_map(
                    batch_gpu["obstacles"], locations, description
                )

                pred_gpu = model(sensor_recordings, locations)

                loss = torch.nn.functional.mse_loss(ground_truth, pred_gpu)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                loss = loss.item()

                train_loss_plotter.append(loss)

                writer.add_scalar("training loss", loss, global_iteration)
                # for k in loss_terms.keys():
                #     writer.add_scalar(k, loss_terms[k].item(), global_iteration)

                progress_bar((global_iteration) % args.plotinterval, args.plotinterval)

                if ((global_iteration + 1) % args.validationinterval) == 0:
                    print("Computing validation loss...")
                    # curr_val_loss = validation_loss(model)
                    curr_val_loss = np.nan
                    if curr_val_loss < best_val_loss:
                        best_val_loss = curr_val_loss
                        if not args.nosave:
                            model.save(make_model_filename("best"))

                    if not args.nosave:
                        model.save(make_model_filename("latest"))

                    val_loss_x.append(global_iteration)
                    val_loss_y.append(curr_val_loss)

                    writer.add_scalar(
                        "validation loss", curr_val_loss, global_iteration
                    )

                time_to_plot = ((global_iteration + 1) % args.plotinterval) == 0 or (
                    global_iteration < 2
                )

                if time_to_plot:
                    print(
                        "Epoch {}, iteration {} of {} ({} %), loss={}".format(
                            e,
                            i,
                            len(train_loader),
                            100 * i // len(train_loader),
                            loss,
                        )
                    )

                if time_to_plot:
                    val_batch_cpu = next(iter(val_loader))
                    val_batch_gpu = val_batch_cpu.to(the_device)

                    # clear figures for a new update
                    ax_t1.cla()
                    ax_b1.cla()
                    ax_t2.cla()
                    ax_b2.cla()
                    ax_t3.cla()
                    ax_b3.cla()
                    ax_t4.cla()
                    ax_b4.cla()

                    # plt.gcf().suptitle(args.experiment)

                    # plot the input waveforms
                    ax_t1.title.set_text("Input (train)")
                    # plot_inputs(ax_t1, batch_cpu, description)
                    ax_b1.title.set_text("Input (validation)")
                    # plot_inputs(ax_b1, val_batch_cpu, description)

                    # plot the ground truth obstacles
                    ax_t2.title.set_text("Ground Truth (train)")
                    plot_ground_truth(ax_t2, batch_cpu["obstacles"][0], description)
                    ax_b2.title.set_text("Ground Truth (validation)")
                    plot_ground_truth(ax_b2, val_batch_cpu["obstacles"][0], description)

                    # plot the predicted sdf
                    ax_t3.title.set_text("Prediction (train)")
                    plot_prediction(
                        ax_t3,
                        model,
                        batch_gpu["sensor_recordings"][0, sensor_indices],
                        description,
                    )
                    ax_b3.title.set_text("Prediction (validation)")
                    plot_prediction(
                        ax_b3,
                        model,
                        val_batch_gpu["sensor_recordings"][0, sensor_indices],
                        description,
                    )

                    # plot the training loss on a log plot
                    ax_t4.title.set_text("Training Loss")
                    train_loss_plotter.plot_to(ax_t4)
                    ax_t4.set_yscale("log")

                    # plot the validation loss on a log plot
                    ax_b4.title.set_text("Validation Loss")
                    ax_b4.set_yscale("log")
                    ax_b4.plot(val_loss_x, val_loss_y, c="Red")

                    # Note: calling show or pause will cause a bad time
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    plt_screenshot(fig).save(
                        log_path + "/image_" + str(global_iteration + 1) + ".png"
                    )

                    if (
                        args.iterations is not None
                        and global_iteration > args.iterations
                    ):
                        print("Done - desired iteration count was reached")
                        return

                global_iteration += 1

        plt.close("all")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # I don't wanna see any stack traces
        pass
