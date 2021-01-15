import os
import fix_dead_command_line

from dataset_config import EmitterConfig, InputConfig, OutputConfig, ReceiverConfig, TrainingConfig
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import datetime
import PIL.Image
import random

from device_dict import DeviceDict
from dataset import WaveSimDataset
from progress_bar import progress_bar
from featurize import make_sdf_image_gt, make_sdf_image_pred, make_heatmap_image_gt, make_heatmap_image_pred, make_depthmap_gt, make_depthmap_pred, make_deterministic_validation_batches_implicit, red_white_blue_banded, red_white_blue
from custom_collate_fn import custom_collate

from EchoLearnNN import EchoLearnNN
from SimpleNN import SimpleNN

to_tensor = torchvision.transforms.ToTensor()

def plt_screenshot():
    fig = plt.gcf()
    pil_img = PIL.Image.frombytes(
        'RGB',
        fig.canvas.get_width_height(),
        fig.canvas.tostring_rgb()
    )
    return pil_img
    # return to_tensor(pil_img)

def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment", type=str, dest="experiment", required=True)
    parser.add_argument("--dataset", type=str, dest="dataset", default="v8")
    parser.add_argument("--maxexamples", type=int, dest="maxexamples", default=None)
    parser.add_argument("--batchsize", type=int, dest="batchsize", default=4)
    parser.add_argument("--receivercount", type=int, choices=[1, 2, 4, 8, 16, 32, 64], dest="receivercount", default=8)
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
    parser.add_argument("--nninput", type=str, dest="nninput", choices=["audioraw", "audiowaveshaped", "spectrogram"], required=True)
    parser.add_argument("--nnoutput", type=str, dest="nnoutput", choices=["sdf", "heatmap", "depthmap"], required=True)
    parser.add_argument("--simplenn", dest="simplenn", default=False, action="store_true")
    parser.add_argument("--plotinterval", type=int, dest="plotinterval", default=32)
    parser.add_argument("--validationinterval", type=int, dest="validationinterval", default=256)

    args = parser.parse_args()

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
        receiver_config=receiver_config
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

    what_my_gpu_can_handle = 128**2

    if (args.nosave):
        print("NOTE: networks are not being saved")

    ecds = WaveSimDataset(
        training_config,
        input_config,
        output_config,
        emitter_config,
        receiver_config
    )

    val_ratio = 1 / 8
    val_size = int(len(ecds)*val_ratio)
    indices_val = list(range(0, val_size))
    indices_train = list(range(val_size, len(ecds)))

    val_set   = torch.utils.data.Subset(ecds, indices_val)
    train_set = torch.utils.data.Subset(ecds, indices_train)

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

    def bogStandardLoss(batch_gt, batch_pred):
        x1 = batch_gt["output"]
        x2 = batch_pred["output"][:, 0]
        mse = torch.nn.functional.mse_loss(x1, x2)
        terms = { "mean_squared_error": mse }
        return mse, terms

    def meanVarianceLoss(batch_gt, batch_pred):
        y = batch_gt["output"]
        z_hat = batch_pred["output"]
        y_hat = z_hat[:, 0]
        sigma_hat_inverse = z_hat[:, 1]

        sqrt2pi = math.sqrt(2.0 * np.pi)
        squared_error = (y - y_hat)**2

        #     phi(y|x)  = exp(-(y - y_hat)^2/(2*sigma^2))
        #                     / (sqrt(2*pi)*sigma)
        # log(phi(y|x)) = -(y - y_hat)^2 / (2*sigma^2)
        #                     - log(sqrt(2*pi)*sigma)
        #               = -0.5 * (y - y_hat)^2 / sigma^2
        #                     - (log(sqrt(2*pi)) + log(sigma))
        #               = -0.5 * (y - y_hat)^2 * (1/sigma)^2
        #                     - (log(sqrt(2*pi)) - log(1/sigma))


        log_numerator = -0.5 * squared_error * sigma_hat_inverse**2
        log_denominator = math.log(sqrt2pi) - torch.log(sigma_hat_inverse)
        log_phi = log_numerator - log_denominator
        nll = torch.mean(-log_phi)
        terms = {
            "mean_squared_error": torch.mean(squared_error).detach(),
            "mean_predicted_variance": torch.mean(1.0/sigma_hat_inverse).detach(),
            "negative_log_likelihood": nll.detach()
        }
        return nll, terms


    
    loss_function = meanVarianceLoss if args.predictvariance else bogStandardLoss
    
    def validation_loss(model):
        print("Computing validation loss (MSE)...")
        losses = []
        for i, batch in enumerate(val_loader):
            if args.implicitfunction:
                num_splits = args.resolution**2 * args.batchsize // what_my_gpu_can_handle
                batches = make_deterministic_validation_batches_implicit(batch, args.nnoutput, args.resolution, num_splits)
                
                losses_batch = []
                for b in batches:
                    b = b.to("cuda")
                    pred = model(b)
                    pred = pred["output"][:, 0]
                    gt = b["output"]
                    loss = torch.nn.functional.mse_loss(pred, gt).detach()
                    losses_batch.append(loss.item())
                losses.append(np.mean(np.asarray(losses_batch)))
            else:
                batch = batch.to("cuda")
                pred = model(batch)
                pred = pred["output"][:, 0]
                gt = batch["output"]
                loss = torch.nn.functional.mse_loss(pred, gt).detach()
                losses.append(loss.item())
            progress_bar(i, len(val_loader))
        return np.mean(np.asarray(losses))

    NetworkType = SimpleNN if args.simplenn else EchoLearnNN

    network = NetworkType(
        input_config=input_config,
        output_config=output_config
    ).cuda()

    model_path = os.environ.get("TRAINING_MODEL_PATH")

    if model_path is None or not os.path.exists(model_path):
        raise Exception("Please set the TRAINING_MODEL_PATH environment variable to point to the desired model directory")

    def save_network(filename):
        path = os.path.join(model_path, filename)
        print(f"Saving model to \"{path}\"")
        torch.save(network.state_dict(), path)

    # def restore_network(filename):
    #     filename = "models/" + filename
    #     print("Loading model from \"{}\"".format(filename))
    #     network.load_state_dict(torch.load(filename))
    #     network.eval()

    def plot_inputs(plt_axis, batch):
        the_input = batch['input'][0].detach()
        if (len(the_input.shape) == 2):
            plt_axis.set_ylim(-1, 1)
            for j in range(args.receivers):
                plt_axis.plot(the_input[j].detach())
        else:
            the_input_min = torch.min(the_input)
            the_input_max = torch.max(the_input)
            the_input = (the_input - the_input_min) / (the_input_max - the_input_min)
            spectrogram_img_grid = torchvision.utils.make_grid(
                the_input.unsqueeze(1).repeat(1, 3, 1, 1),
                nrow=1
            )
            plt_axis.imshow(spectrogram_img_grid.permute(1, 2, 0))
            plt_axis.axis("off")

    def plot_ground_truth(plt_axis, batch, show_samples=False):
        if args.nnoutput == "sdf":
            img = make_sdf_image_gt(batch, args.resolution).cpu()
            plt_axis.imshow(red_white_blue_banded(img), interpolation="bicubic")
            plt_axis.axis("off")
            if show_samples and args.implicitfunction:
                yx = batch["params"][0].detach() * args.resolution
                plt_axis.scatter(yx[:,1], yx[:,0], s=1.0)
        elif args.nnoutput == "heatmap":
            img = make_heatmap_image_gt(batch, args.resolution).cpu()
            plt_axis.imshow(red_white_blue(img), interpolation="bicubic")
            plt_axis.axis("off")
            if show_samples and args.implicitfunction:
                yx = batch["params"][0].detach() * args.resolution
                plt_axis.scatter(yx[:,1], yx[:,0], s=1.0)
        elif args.nnoutput == "depthmap":
            arr = make_depthmap_gt(batch, args.resolution).cpu()
            plt_axis.set_ylim(-0.5, 1.5)
            plt_axis.plot(arr)
            # TODO: show samples somehow (maybe by dots along gt arr)
        else:
            raise Exception("Unrecognized output representation")

    def plot_image(plt_axis, img, display_fn):
        y = img[0]
        assert(y.shape == (args.resolution, args.resolution))
        plt_axis.imshow(display_fn(y), interpolation="bicubic")
        if (args.predictvariance):
            sigma = 1.0 / img[1]
            assert(sigma.shape == (args.resolution, args.resolution))
            sigma_clamped = torch.clamp(sigma, 0.0, 1.0)
            gamma_value = 0.5
            sigma_curved = sigma_clamped**gamma_value
            mask = torch.cat((
                torch.zeros((args.resolution, args.resolution, 3)),
                sigma_curved.unsqueeze(-1),
            ), dim=2)
            plt_axis.imshow(mask, interpolation="bicubic")
        plt_axis.axis("off")

    def plot_depthmap(plt_axis, data):
        y = data[0]
        assert(y.shape == (args.resolution,))
        plt_axis.set_ylim(-0.5, 1.5)
        plt_axis.plot(y, c="black")
        if (args.predictvariance):
            sigma = 1.0 / data[1]
            assert(sigma.shape == (args.resolution,))
            plt_axis.plot(y - sigma, c="red")
            plt_axis.plot(y + sigma, c="red")

    def plot_prediction(plt_axis, batch, network):
        if args.implicitfunction:
            if args.nnoutput == "sdf":
                num_splits = args.resolution**2 // what_my_gpu_can_handle
                img = make_sdf_image_pred(batch, args.resolution, network, num_splits, args.predictvariance)
                plot_image(plt_axis, img, red_white_blue_banded)
            elif args.nnoutput == "heatmap":
                num_splits = args.resolution**2 // what_my_gpu_can_handle
                img = make_heatmap_image_pred(batch, args.resolution, network, num_splits, args.predictvariance)
                plot_image(plt_axis, img, red_white_blue)
            elif args.nnoutput == "depthmap":
                arr = make_depthmap_pred(batch, args.resolution, network)
                plot_depthmap(plt_axis, arr)
            else:
                raise Exception("Unrecognized output representation")
        else:
            # non-implicit function
            output = network(batch)['output'][0].detach().cpu()
            if args.nnoutput == "sdf":
                plot_image(plt_axis, output, red_white_blue_banded)
            elif args.nnoutput == "heatmap":
                plot_image(plt_axis, output, red_white_blue)
            elif args.nnoutput == "depthmap":
                plot_depthmap(plt_axis, output)
            else:
                raise Exception("Unrecognized output representation")

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

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
        ax_b1 = axes[1,0]
        ax_b2 = axes[1,1]
        ax_b3 = axes[1,2]
        ax_t4 = axes[0,3]
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
                batch_gpu = batch_cpu.to('cuda')

                pred_gpu = network(batch_gpu)
                pred_cpu = pred_gpu.to('cpu')
                
                loss, loss_terms = loss_function(batch_gpu, pred_gpu)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                writer.add_scalar("training loss", loss.item(), global_iteration)
                for k in loss_terms.keys():
                    writer.add_scalar(k, loss_terms[k].item(), global_iteration)
                
                progress_bar((global_iteration) % args.plotinterval, args.plotinterval)

                if ((global_iteration + 1) % args.validationinterval) == 0:
                    curr_val_loss = validation_loss(network)
                    if curr_val_loss < best_val_loss:
                        best_val_loss = curr_val_loss
                        if not args.nosave:
                            save_network("{}_{}_model_best.dat".format(args.experiment, timestamp))
                            
                    if not args.nosave:
                        save_network("{}_{}_model_latest.dat".format(args.experiment, timestamp))


                    val_loss_x.append(len(losses))
                    val_loss_y.append(curr_val_loss)
                    
                    writer.add_scalar("validation loss", curr_val_loss, global_iteration)

                time_to_plot = ((global_iteration + 1) % args.plotinterval) == 0
                time_to_save_figure = ((global_iteration + 1) % (args.plotinterval * 8)) == 0

                if time_to_plot:
                    print("Epoch {}, iteration {} of {} ({} %), loss={}".format(e, i, len(train_loader), 100*i//len(train_loader), losses[-1]))

                if time_to_plot and (not args.nodisplay or time_to_save_figure):
                    val_batch_cpu = next(iter(val_loader))
                    val_batch_gpu = val_batch_cpu.to('cuda')

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
                    plot_inputs(ax_t1, batch_cpu)
                    ax_b1.title.set_text("Input (validation)")
                    plot_inputs(ax_b1, val_batch_cpu)
                    
                    # plot the ground truth obstacles
                    ax_t2.title.set_text("Ground Truth (train)")
                    plot_ground_truth(ax_t2, batch_cpu, show_samples=False)
                    ax_b2.title.set_text("Ground Truth (validation)")
                    plot_ground_truth(ax_b2, val_batch_cpu)
                    
                    # plot the predicted sdf
                    ax_t3.title.set_text("Prediction (train)")
                    plot_prediction(ax_t3, batch_gpu, network)
                    ax_b3.title.set_text("Prediction (validation)")
                    plot_prediction(ax_b3, val_batch_gpu, network)
                    
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

                    if time_to_save_figure:
                        plt_screenshot().save(log_path + "/image_" + str(global_iteration + 1) + ".png")

                        # NOTE: this is done in the screenshot branch so that a screenshot with the final
                        # performance is always included
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
