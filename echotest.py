import fix_dead_command_line

import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from dataset_config import EmitterConfig, InputConfig, OutputConfig, ReceiverConfig, TrainingConfig

def main():
    parser = ArgumentParser()
    # TODO: remove all defaults affecting network structure
    parser.add_argument("--description", type=str, dest="description", required=True)
    parser.add_argument("--batchsize", type=int, dest="batchsize", default=4)
    parser.add_argument("--nodisplay", dest="nodisplay", default=False, action="store_true")

    parser.add_argument("--receivercount", type=int, choices=[1, 2, 4, 8, 16, 32, 64], dest="receivercount")
    parser.add_argument("--receiverarrangement", type=str, choices=["flat", "grid"], dest="receiverarrangement")
    parser.add_argument("--emitterarrangement", type=str, choices=["mono", "stereo", "surround"], dest="emitterarrangement")
    parser.add_argument("--emittersignal", type=str, choices=["impulse", "beep", "sweep"], dest="emittersignal")
    parser.add_argument("--emittersequential", type=bool, dest="emittersequential")
    parser.add_argument("--emittersamefrequency", type=bool, dest="emittersamefrequency")
    parser.add_argument("--implicitfunction", type=bool, dest="implicitfunction")
    parser.add_argument("--predictvariance", type=bool, dest="predictvariance")
    parser.add_argument("--resolution", type=int, dest="resolution")
    parser.add_argument("--nninput", type=str, dest="nninput", choices=["audioraw", "audiowaveshaped", "spectrogram"], required=True)
    parser.add_argument("--nnoutput", type=str, dest="nnoutput", choices=["sdf", "heatmap", "depthmap"], required=True)
    parser.add_argument("--summarystatistics", type=bool, dest="summarystatistics")

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
        max_examples=args.maxexamples,
        max_obstacles=args.maxobstacles,
        circles_only=args.circlesonly,
        allow_occlusions=args.allowocclusions,
        importance_sampling=args.importancesampling,
        samples_per_example=args.samplesperexample
    )