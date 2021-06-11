import fix_dead_command_line

import pickle
import torch
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
import numpy as np

from featurize import make_dense_implicit_output_pred
from custom_collate_fn import custom_collate
from device_dict import DeviceDict
from dataset import WaveSimDataset
from Echo4ChDataset import Echo4ChDataset
from config import (
    EmitterConfig,
    InputConfig,
    OutputConfig,
    ReceiverConfig,
    TrainingConfig,
)
from config_constants import (
    input_format_all,
    receiver_arrangement_all,
    emitter_format_all,
    emitter_arrangement_all,
    output_format_all,
    output_format_depthmap,
)
from EchoLearnNN import EchoLearnNN
from the_device import the_device
from progress_bar import progress_bar
from visualization import plot_ground_truth, plot_prediction, plt_screenshot
from realbool import realbool
from convert_output import (
    compute_error_metrics,
    ground_truth_occupancy,
    predicted_occupancy,
)


def main():
    parser = ArgumentParser()
    parser.add_argument("--description", type=str, dest="description", required=True)
    # parser.add_argument("--batchsize", type=int, dest="batchsize", default=4)
    parser.add_argument(
        "--nodisplay", dest="nodisplay", default=False, action="store_true"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wavesim", "echo4ch"],
        dest="dataset",
        default="wavesim",
    )

    parser.add_argument(
        "--receivercount",
        type=int,
        choices=[1, 2, 4, 8, 16, 32, 64],
        dest="receivercount",
        required=True,
    )
    parser.add_argument(
        "--receiverarrangement",
        type=str,
        choices=receiver_arrangement_all,
        dest="receiverarrangement",
        required=True,
    )
    parser.add_argument(
        "--emitterarrangement",
        type=str,
        choices=emitter_arrangement_all,
        dest="emitterarrangement",
        required=True,
    )
    parser.add_argument(
        "--emittersignal",
        type=str,
        choices=emitter_format_all,
        dest="emittersignal",
        required=True,
    )
    parser.add_argument(
        "--emittersequential", type=realbool, dest="emittersequential", required=True
    )
    parser.add_argument(
        "--emittersamefrequency",
        type=realbool,
        dest="emittersamefrequency",
        required=True,
    )
    parser.add_argument(
        "--implicitfunction", type=realbool, dest="implicitfunction", required=True
    )
    parser.add_argument(
        "--predictvariance", type=realbool, dest="predictvariance", required=True
    )
    parser.add_argument("--resolution", type=int, dest="resolution", required=True)
    parser.add_argument(
        "--nninput",
        type=str,
        dest="nninput",
        choices=input_format_all,
        required=True,
    )
    parser.add_argument(
        "--nnoutput",
        type=str,
        dest="nnoutput",
        choices=output_format_all,
        required=True,
    )
    parser.add_argument(
        "--summarystatistics", type=realbool, dest="summarystatistics", required=True
    )

    parser.add_argument("--modelpath", type=str, dest="modelpath", required=True)

    parser.add_argument(
        "--makeimages", dest="makeimages", default=False, action="store_true"
    )
    parser.add_argument(
        "--computemetrics", dest="computemetrics", default=False, action="store_true"
    )
    parser.add_argument(
        "--occupancyshadows",
        dest="occupancyshadows",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--tofcropping", type=realbool, dest="tofcropping", required=True
    )
    parser.add_argument(
        "--tofwindowsize",
        type=int,
        dest="tofwindowsize",
        choices=[64, 128, 256, 512, 1024],
        default=None,
    )

    args = parser.parse_args()

    assert args.tofcropping == (args.tofwindowsize is not None)

    dataset_name = args.dataset
    using_echo4ch = dataset_name == "echo4ch"

    if args.nodisplay:
        matplotlib.use("Agg")

    emitter_config = EmitterConfig(
        arrangement=args.emitterarrangement,
        format=args.emittersignal,
        sequential=args.emittersequential,
        sameFrequency=args.emittersamefrequency,
    )

    receiver_config = ReceiverConfig(
        arrangement=args.receiverarrangement, count=args.receivercount
    )

    input_config = InputConfig(
        format=args.nninput,
        emitter_config=emitter_config,
        receiver_config=receiver_config,
        summary_statistics=args.summarystatistics,
        tof_crop_size=args.tofwindowsize,
        using_echo4ch=using_echo4ch,
    )

    output_config = OutputConfig(
        format=args.nnoutput,
        implicit=args.implicitfunction,
        predict_variance=args.predictvariance,
        tof_cropping=args.tofcropping,
        resolution=args.resolution,
        using_echo4ch=using_echo4ch,
    )

    training_config = TrainingConfig(
        max_examples=None,
        max_obstacles=None,
        circles_only=False,
        allow_occlusions=True,
        importance_sampling=False,
        samples_per_example=128,
    )

    shadows = args.occupancyshadows

    # If not occupancy shadows, then not depthmap
    assert shadows or (output_config.format != output_format_depthmap)

    print("============== CONFIGURATIONS ==============")
    print(f"Dataset: {args.dataset}")
    emitter_config.print()
    receiver_config.print()
    input_config.print()
    output_config.print()
    # training_config.print()
    print("============================================")
    print("")

    if dataset_name == "wavesim":
        ds = WaveSimDataset(
            training_config,
            input_config,
            output_config,
            emitter_config,
            receiver_config,
        )
    elif dataset_name == "echo4ch":
        ds = Echo4ChDataset(
            training_config,
            input_config,
            output_config,
            emitter_config,
            receiver_config,
        )
    else:
        raise Exception(f'Unrecognized dataset type "{dataset_name}"')

    collate_fn_device = lambda batch: DeviceDict(custom_collate(batch))
    test_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn_device,
    )

    network = EchoLearnNN(input_config, output_config)
    network.restore(args.modelpath)
    network = network.to(the_device)

    plt.ion()
    nrows = 2 if args.computemetrics else 1
    fig, ax = plt.subplots(nrows, 2, figsize=(8, 4 * nrows), dpi=64)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))

    if args.computemetrics:
        ax_l = ax[0, 0]
        ax_r = ax[0, 1]
        ax_bl = ax[1, 0]
        ax_br = ax[1, 1]
    else:
        ax_l = ax[0]
        ax_r = ax[1]
        ax_bl = None
        ax_br = None

    def plot_occupancy(plt_axis, occupancy):
        assert isinstance(occupancy, torch.BoolTensor) or isinstance(
            occupancy, torch.cuda.BoolTensor
        )
        plt_axis.imshow(occupancy[0, 0, :, :].cpu().float())

    N = len(test_loader)

    if args.makeimages:
        if not os.path.exists("images"):
            os.makedirs("images")

    if args.computemetrics:
        if not os.path.exists("metrics"):
            os.makedirs("metrics")

    total_metrics = {}

    def aggregrate_metrics(metrics):
        assert isinstance(metrics, dict)
        for k, v in metrics.items():
            assert isinstance(v, float)
            if not k in total_metrics:
                total_metrics[k] = []
            total_metrics[k].append(v)

    def report_metrics():
        print("=================== Error Metrics ===================")
        print(f"Occupancy shadows? {'Yes' if shadows else 'No'}")
        for k, v in total_metrics.items():
            assert isinstance(v, list)
            arr = np.array(v)
            print(f"{k} (n={len(arr)}):")
            print(f"  mean               : {np.mean(arr)}")
            print(f"  variance           : {np.var(arr)}")
            print(f"  standard deviation : {np.std(arr)}")
            print(f"  minumum            : {np.min(arr)}")
            print(f"  maximum            : {np.max(arr)}")
        print("=====================================================")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(the_device)
            ax_l.cla()
            ax_r.cla()
            plot_ground_truth(ax_l, batch, output_config)
            plot_prediction(ax_r, batch, network, output_config)

            if args.computemetrics:
                ax_bl.cla()
                ax_br.cla()
                obstacles = batch["obstacles_list"][0]
                occupancy_gt = ground_truth_occupancy(
                    obstacles, output_config.resolution, shadows
                )
                if output_config.implicit:
                    pred = make_dense_implicit_output_pred(
                        batch, network, output_config
                    )
                    pred = pred.unsqueeze(0).to(the_device)
                else:
                    pred = network(batch)["output"]
                occupancy_pred = predicted_occupancy(pred, output_config, shadows)
                metrics = compute_error_metrics(occupancy_gt, occupancy_pred)
                aggregrate_metrics(metrics)

                plot_occupancy(ax_bl, occupancy_gt)
                plot_occupancy(ax_br, occupancy_pred)

            plt.gcf().canvas.draw()
            plt.gcf().canvas.flush_events()

            if args.makeimages:
                plt_screenshot(fig).save(f"images/{args.description}_{i}.png")
            progress_bar(i, N)

        if args.computemetrics:
            report_metrics()
            with open(f"metrics/metrics_{args.description}.pkl", "wb") as outfile:
                pickle.dump(total_metrics, outfile)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # I don't wanna see any stack traces
        pass
