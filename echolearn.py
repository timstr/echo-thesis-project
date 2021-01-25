import fix_dead_command_line

from visualization import plot_ground_truth, plot_inputs, plot_prediction, plt_screenshot
import os
from loss_functions import compute_loss_on_dataset, meanAndVarianceLoss, meanSquaredErrorLoss
from dataset_config import EmitterConfig, InputConfig, OutputConfig, ReceiverConfig, TrainingConfig
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime

from the_device import the_device
from device_dict import DeviceDict
from dataset import WaveSimDataset
from Echo4ChDataset import Echo4ChDataset
from progress_bar import progress_bar
from custom_collate_fn import custom_collate

from EchoLearnNN import EchoLearnNN

to_tensor = torchvision.transforms.ToTensor()

def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment", type=str, dest="experiment", required=True)
    parser.add_argument("--maxexamples", type=int, dest="maxexamples", default=None)
    parser.add_argument("--batchsize", type=int, dest="batchsize", default=4)
    parser.add_argument("--receivercount", type=int, choices=[1, 2, 4, 8, 16, 32, 64], dest="receivercount", default=8)
    parser.add_argument("--dataset", type=str, choices=["wavesim", "echo4ch"], dest="dataset", default="wavesim")
    parser.add_argument("--receiverarrangement", type=str, choices=["flat", "grid"], dest="receiverarrangement", default="grid")
    parser.add_argument("--emitterarrangement", type=str, choices=["mono", "stereo", "surround"], dest="emitterarrangement", default="mono")
    parser.add_argument("--emittersignal", type=str, choices=["impulse", "beep", "sweep"], dest="emittersignal", default="sweep")
    parser.add_argument("--emittersequential", dest="emittersequential", default=False, action="store_true")
    parser.add_argument("--emittersamefrequency", dest="emittersamefrequency", default=False, action="store_true")
    parser.add_argument("--maxobstacles", type=int, dest="maxobstacles", default=None)
    parser.add_argument("--circlesonly", dest="circlesonly", default=False, action="store_true")
    parser.add_argument("--allowocclusions", dest="allowocclusions", default=True, action="store_true")
    parser.add_argument("--nosave", dest="nosave", default=False, action="store_true")
    parser.add_argument("--nodisplay", dest="nodisplay", default=False, action="store_true")
    parser.add_argument("--iterations", type=int, dest="iterations", default=None)
    parser.add_argument("--implicitfunction", dest="implicitfunction", default=False, action="store_true")
    parser.add_argument("--samplesperexample", type=int, dest="samplesperexample", default=128)
    parser.add_argument("--importancesampling", dest="importancesampling", default=False, action="store_true")
    parser.add_argument("--predictvariance", dest="predictvariance", default=False, action="store_true")
    parser.add_argument("--resolution", type=int, dest="resolution", default=128)
    parser.add_argument("--nninput", type=str, dest="nninput", choices=["audioraw", "audiowaveshaped", "spectrogram", "gccphat"], required=True)
    parser.add_argument("--nnoutput", type=str, dest="nnoutput", choices=["sdf", "heatmap", "depthmap"], required=True)
    parser.add_argument("--summarystatistics", dest="summarystatistics", default=False, action="store_true")
    parser.add_argument("--plotinterval", type=int, dest="plotinterval", default=32)
    parser.add_argument("--validationinterval", type=int, dest="validationinterval", default=256)
    parser.add_argument("--restoremodelpath", type=str, dest="restoremodelpath", default=None)

    args = parser.parse_args()

    if args.nodisplay:
        matplotlib.use("Agg")

    dataset_name = args.dataset
    using_echo4ch = (dataset_name == "echo4ch")

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
        emitter_config=emitter_config,
        receiver_config=receiver_config,
        summary_statistics=args.summarystatistics,
        using_echo4ch=using_echo4ch
    )

    output_config = OutputConfig(
        format=args.nnoutput,
        implicit=args.implicitfunction,
        predict_variance=args.predictvariance,
        resolution=args.resolution,
        using_echo4ch=using_echo4ch
    )

    training_config = TrainingConfig(
        max_examples=args.maxexamples,
        max_obstacles=args.maxobstacles,
        circles_only=args.circlesonly,
        allow_occlusions=args.allowocclusions,
        importance_sampling=args.importancesampling,
        samples_per_example=args.samplesperexample
    )


    print("============== CONFIGURATIONS ==============")
    print(f"Dataset: {args.dataset}")
    emitter_config.print()
    receiver_config.print()
    input_config.print()
    output_config.print()
    training_config.print()
    print("============================================")
    print("")

    if (args.nosave):
        print("NOTE: networks are not being saved")

    network = EchoLearnNN(
        input_config=input_config,
        output_config=output_config
    ).to(the_device)

    if args.restoremodelpath is not None:
        network.restore(args.restoremodelpath)

    if dataset_name == "wavesim":
        ds = WaveSimDataset(
            training_config,
            input_config,
            output_config,
            emitter_config,
            receiver_config
        )
    elif dataset_name == "echo4ch":
        ds = Echo4ChDataset(
            training_config,
            input_config,
            output_config,
            emitter_config,
            receiver_config
        )
    else:
        raise Exception(f"Unrecognized dataset type \"{dataset_name}\"")

    val_ratio = 0.1
    val_size = int(len(ds)*val_ratio)
    indices_val = list(range(0, val_size))
    indices_train = list(range(val_size, len(ds)))

    val_set   = torch.utils.data.Subset(ds, indices_val)
    train_set = torch.utils.data.Subset(ds, indices_train)

    # define the dataset loader (batch size, shuffling, ...)
    collate_fn_device = lambda batch : DeviceDict(custom_collate(batch))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batchsize,
        num_workers=0,
        pin_memory=False, # Note, setting pin_memory=False to avoid the pin_memory call
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_device
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batchsize,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_device
    )

    loss_function = meanAndVarianceLoss if output_config.predict_variance else meanSquaredErrorLoss

    def validation_loss(the_network):
        return compute_loss_on_dataset(the_network, val_loader, meanSquaredErrorLoss, output_config)

    model_path = os.environ.get("TRAINING_MODEL_PATH")

    if model_path is None or not os.path.exists(model_path):
        raise Exception("Please set the TRAINING_MODEL_PATH environment variable to point to the desired model directory")     

    def make_model_filename(label):
        assert isinstance(label, str)
        fname = f"{args.experiment}_{timestamp}_{label}.dat"
        return os.path.join(model_path, fname)

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-10)
    maximum_gradient_norm = 20.0

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    log_path_root = os.environ.get("TRAINING_LOG_PATH")

    if log_path_root is None or not os.path.exists(log_path_root):
        raise Exception("Please set the TRAINING_LOG_PATH environment variable to point to the desired log directory")

    log_folder_name = f"{args.experiment}_{timestamp}"

    log_path = os.path.join(log_path_root, log_folder_name)
    
    with SummaryWriter(log_path) as writer:

        if not args.nodisplay:
            plt.ion()

        fig, axes = plt.subplots(2, 4, figsize=(22, 10), dpi=80)
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))

        ax_t1 = axes[0,0]
        ax_t2 = axes[0,1]
        ax_t3 = axes[0,2]
        ax_t4 = axes[0,3]
        ax_b1 = axes[1,0]
        ax_b2 = axes[1,1]
        ax_b3 = axes[1,2]
        ax_b4 = axes[1,3]

        num_epochs = 1000000
        losses = []
        val_loss_y = []
        val_loss_x = []
        best_val_loss = np.inf
        global_iteration = 0
        for e in range(num_epochs):
            train_iter = iter(train_loader)
            for i in range(len(train_loader)):
                batch_cpu = next(train_iter)
                batch_gpu = batch_cpu.to(the_device)

                pred_gpu = network(batch_gpu)
                
                loss, loss_terms = loss_function(batch_gpu, pred_gpu)
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(network.parameters(), maximum_gradient_norm)

                optimizer.step()
                losses.append(loss.item())

                writer.add_scalar("training loss", loss.item(), global_iteration)
                for k in loss_terms.keys():
                    writer.add_scalar(k, loss_terms[k].item(), global_iteration)
                
                progress_bar((global_iteration) % args.plotinterval, args.plotinterval)

                if ((global_iteration + 1) % args.validationinterval) == 0:
                    print("Computing validation loss...")
                    curr_val_loss = validation_loss(network)
                    if curr_val_loss < best_val_loss:
                        best_val_loss = curr_val_loss
                        if not args.nosave:
                            network.save(make_model_filename("best"))
                            
                    if not args.nosave:
                        network.save(make_model_filename("latest"))


                    val_loss_x.append(len(losses))
                    val_loss_y.append(curr_val_loss)
                    
                    writer.add_scalar("validation loss", curr_val_loss, global_iteration)

                time_to_plot = ((global_iteration + 1) % args.plotinterval) == 0

                if time_to_plot:
                    print("Epoch {}, iteration {} of {} ({} %), loss={}".format(e, i, len(train_loader), 100*i//len(train_loader), losses[-1]))

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
                    plot_inputs(ax_t1, batch_cpu, receiver_config)
                    ax_b1.title.set_text("Input (validation)")
                    plot_inputs(ax_b1, val_batch_cpu, receiver_config)
                    
                    # plot the ground truth obstacles
                    ax_t2.title.set_text("Ground Truth (train)")
                    plot_ground_truth(ax_t2, batch_cpu, output_config, show_samples=False)
                    ax_b2.title.set_text("Ground Truth (validation)")
                    plot_ground_truth(ax_b2, val_batch_cpu, output_config)
                    
                    # plot the predicted sdf
                    ax_t3.title.set_text("Prediction (train)")
                    plot_prediction(ax_t3, batch_gpu, network, output_config)
                    ax_b3.title.set_text("Prediction (validation)")
                    plot_prediction(ax_b3, val_batch_gpu, network, output_config)
                    
                    # plot the training loss on a log plot
                    ax_t4.title.set_text("Training Loss")
                    ax_t4.scatter(range(len(losses)), losses, s=1.0)
                    if not args.predictvariance:
                        ax_t4.set_yscale('log')

                    # plot the validation loss on a log plot
                    ax_b4.title.set_text("Validation Loss")
                    ax_b4.set_yscale('log')
                    ax_b4.plot(val_loss_x, val_loss_y, c="Red")

                    # Note: calling show or pause will cause a bad time
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    plt_screenshot(fig).save(log_path + "/image_" + str(global_iteration + 1) + ".png")

                    if args.iterations is not None and global_iteration > args.iterations:
                        print("Done - desired iteration count was reached")
                        return

                global_iteration += 1

        plt.close('all')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # I don't wanna see any stack traces
        pass
