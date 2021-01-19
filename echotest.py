import fix_dead_command_line

import torch
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os

from custom_collate_fn import custom_collate
from device_dict import DeviceDict
from dataset import WaveSimDataset
from dataset_config import EmitterConfig, InputConfig, OutputConfig, ReceiverConfig, TrainingConfig
from EchoLearnNN import EchoLearnNN
from the_device import the_device
from progress_bar import progress_bar
from visualization import plot_ground_truth, plot_prediction, plt_screenshot

def main():
    parser = ArgumentParser()
    # TODO: remove all defaults affecting network structure
    parser.add_argument("--description", type=str, dest="description", required=True)
    # parser.add_argument("--batchsize", type=int, dest="batchsize", default=4)
    parser.add_argument("--nodisplay", dest="nodisplay", default=False, action="store_true")

    parser.add_argument("--receivercount", type=int, choices=[1, 2, 4, 8, 16, 32, 64], dest="receivercount", required=True)
    parser.add_argument("--receiverarrangement", type=str, choices=["flat", "grid"], dest="receiverarrangement", required=True)
    parser.add_argument("--emitterarrangement", type=str, choices=["mono", "stereo", "surround"], dest="emitterarrangement", required=True)
    parser.add_argument("--emittersignal", type=str, choices=["impulse", "beep", "sweep"], dest="emittersignal", required=True)
    parser.add_argument("--emittersequential", type=bool, dest="emittersequential", required=True)
    parser.add_argument("--emittersamefrequency", type=bool, dest="emittersamefrequency", required=True)
    parser.add_argument("--implicitfunction", type=bool, dest="implicitfunction", required=True)
    parser.add_argument("--predictvariance", type=bool, dest="predictvariance", required=True)
    parser.add_argument("--resolution", type=int, dest="resolution", required=True)
    parser.add_argument("--nninput", type=str, dest="nninput", choices=["audioraw", "audiowaveshaped", "spectrogram"], required=True)
    parser.add_argument("--nnoutput", type=str, dest="nnoutput", choices=["sdf", "heatmap", "depthmap"], required=True)
    parser.add_argument("--summarystatistics", type=bool, dest="summarystatistics", required=True)

    parser.add_argument("--modelpath", type=str, dest="modelpath", required=True)

    parser.add_argument("--makeimages", dest="makeimages", default=False, action="store_true")
    parser.add_argument("--computeoccupancyiou", dest="computeoccupancyiou", default=False, action="store_true")
    parser.add_argument("--computeshadowoccupancyiou", dest="computeshadowoccupancyiou", default=False, action="store_true")
    parser.add_argument("--computeprecisionrecall", dest="computeprecisionrecall", default=False, action="store_true")
    
    args = parser.parse_args()

    # TODO: construct config objects (just like in echolearn)
    # TODO: instantiate network
    # TODO: restore network from file
    # TODO: load dataset (from same environment variable as in echolearn)
    # TODO: iterate over dataset, compute desired metrics (save them?) and display/save images

    # Heatmap and SDF can be saved as images without using pyplot
    # depthmap may need pyplot :(

    
    if args.nodisplay:
        matplotlib.use("Agg")

    emitter_config = EmitterConfig(
        arrangement=args.emitterarrangement,
        format=args.emittersignal,
        sequential=args.emittersequential,
        sameFrequency=args.emittersamefrequency
    )

    receiver_config = ReceiverConfig(
        arrangement=args.receiverarrangement,
        count=args.receivercount
    )

    input_config = InputConfig(
        format=args.nninput,
        receiver_config=receiver_config,
        summary_statistics=args.summarystatistics
    )

    output_config = OutputConfig(
        format=args.nnoutput,
        implicit=args.implicitfunction,
        predict_variance=args.predictvariance,
        resolution=args.resolution
    )

    training_config = TrainingConfig(
        max_examples=None,
        max_obstacles=None,
        circles_only=False,
        allow_occlusions=True,
        importance_sampling=False,
        samples_per_example=128
    )

    ecds = WaveSimDataset(
        training_config,
        input_config,
        output_config,
        emitter_config,
        receiver_config
    )

    collate_fn_device = lambda batch : DeviceDict(custom_collate(batch))
    test_loader = torch.utils.data.DataLoader(
        ecds,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn_device
    )

    network = EchoLearnNN(input_config, output_config)
    network.restore(args.modelpath)
    network = network.to(the_device)

    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=64)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    # TODO: plot size and layout

    ax_l = ax[0]
    ax_r = ax[1]

    N = len(test_loader)

    if args.makeimages:
        if not os.path.exists("images"):
            os.makedirs("images")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(the_device)
            ax_l.cla()
            ax_r.cla()
            plot_ground_truth(ax_l, batch, output_config)
            plot_prediction(ax_r, batch, network, output_config)
            plt.gcf().canvas.draw()
            plt.gcf().canvas.flush_events()
            if args.makeimages:
                plt_screenshot(fig).save(f"images/{args.description}_{i}.png")
            progress_bar(i, N)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # I don't wanna see any stack traces
        pass
