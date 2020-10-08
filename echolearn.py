import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import math
import numpy as np
import datetime
import PIL.Image
import random

from device_dict import DeviceDict
from dataset import WaveSimDataset
from progress_bar import progress_bar
from featurize import make_sdf_image_gt, make_sdf_image_pred, make_receiver_indices, make_heatmap_image_gt, make_heatmap_image_pred, make_depthmap_gt, make_depthmap_pred, make_deterministic_validation_batches_implicit, red_white_blue_banded, red_white_blue
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
    parser.add_argument("--numexamples", type=int, dest="numexamples", default=None)
    parser.add_argument("--batchsize", type=int, dest="batchsize", default=4)
    parser.add_argument("--receivers", type=int, choices=[1, 2, 4, 8, 16, 32, 64], dest="receivers", default=8)
    parser.add_argument("--arrangement", type=str, choices=["flat", "grid"], dest="arrangement", default="grid")
    parser.add_argument("--maxobstacles", type=int, dest="maxobstacles", default=None)
    parser.add_argument("--circlesonly", dest="circlesonly", default=False, action="store_true")
    parser.add_argument("--nosave", dest="nosave", default=False, action="store_true")
    parser.add_argument("--iterations", type=int, dest="iterations", default=None)
    parser.add_argument("--implicitfunction", dest="implicitfunction", default=False, action="store_true")
    parser.add_argument("--predictvariance", dest="predictvariance", default=False, action="store_true")
    parser.add_argument("--resolution", type=int, dest="resolution", default=32)
    parser.add_argument("--nninput", type=str, dest="nninput", choices=["audioraw", "audiowaveshaped", "spectrogram"], required=True)
    parser.add_argument("--nnoutput", type=str, dest="nnoutput", choices=["sdf", "heatmap", "depthmap"], required=True)
    parser.add_argument("--simplenn", dest="simplenn", default=False, action="store_true")
    parser.add_argument("--plotinterval", type=int, dest="plotinterval", default=32)
    parser.add_argument("--validationinterval", type=int, dest="validationinterval", default=256)

    args = parser.parse_args()

    receiver_indices = make_receiver_indices(args.receivers, args.arrangement)

    if args.nninput == "audioraw":
        input_format = "1D"
    elif args.nninput == "audiowaveshaped":
        input_format = "1D"
    elif args.nninput == "spectrogram":
        input_format = "2D"

    if args.nnoutput == "sdf":
        implicit_params = 2
        output_format = "2D"
    elif args.nnoutput == "depthmap":
        implicit_params = 1
        output_format = "1D"
    elif args.nnoutput == "heatmap":
        implicit_params = 2
        output_format = "2D"

    if args.implicitfunction:
        output_format = "scalar"
    else:
        implicit_params = 0
    
    what_my_gpu_can_handle = 128*128

    output_dims = 2 if args.predictvariance else 1

    if (args.nosave):
        print("NOTE: networks are not being saved")

    ecds = WaveSimDataset(
        data_folder="dataset/{}".format(args.dataset),
        samples_per_example=128,
        num_examples=args.numexamples,
        max_obstacles=args.maxobstacles,
        receiver_indices=receiver_indices,
        circles_only=args.circlesonly,
        input_representation=args.nninput,
        output_representation=args.nnoutput,
        implicit_function=args.implicitfunction,
        dense_output_resolution=args.resolution
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
        sigma_hat = z_hat[:, 1]
        sumsqrdiff = (y - y_hat)**2
        mse = torch.mean(sumsqrdiff)
        mv = torch.mean(sigma_hat)
        l = torch.mean((sumsqrdiff / (0.001 + sigma_hat)) + sigma_hat)
        terms = { "mean_squared_error": mse, "mean_predicted_variance": mv }
        return l, terms

    
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
        num_input_channels=args.receivers,
        num_implicit_params=implicit_params,
        input_format=input_format,
        output_format=output_format,
        output_resolution=(None if args.implicitfunction else args.resolution),
        predict_variance=args.predictvariance
    ).cuda()

    def save_network(filename):
        filename = "models/" + filename
        print("Saving model to \"{}\"".format(filename))
        torch.save(network.state_dict(), filename)

    def restore_network(filename):
        filename = "models/" + filename
        print("Loading model from \"{}\"".format(filename))
        network.load_state_dict(torch.load(filename))
        network.eval()

    cmap_heatmap = "coolwarm"

    def plot_inputs(plt_axis, batch):
        the_input = batch['input'][0].detach()
        if (len(the_input.shape) == 2):
            for j in range(args.receivers):
                plt_axis.plot(the_input[j].detach())
            plt_axis.set_ylim(-1, 1)
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

    def plot_ground_truth(plt_axis, batch):
        if args.nnoutput == "sdf":
            img = make_sdf_image_gt(batch, args.resolution).cpu()
            plt_axis.imshow(red_white_blue_banded(img), interpolation="bicubic")
            plt_axis.axis("off")
        elif args.nnoutput == "heatmap":
            img = make_heatmap_image_gt(batch, args.resolution).cpu()
            plt_axis.imshow(red_white_blue(img), interpolation="bicubic")
            plt_axis.axis("off")
        elif args.nnoutput == "depthmap":
            arr = make_depthmap_gt(batch, args.resolution).cpu()
            plt_axis.plot(arr)
            plt_axis.set_ylim(-0.5, 1.5)
        else:
            raise Exception("Unrecognized output representation")

    def plot_image(plt_axis, img, display_fn):
        y = img[0]
        assert(y.shape == (args.resolution, args.resolution))
        plt_axis.imshow(display_fn(y), interpolation="bicubic")
        if (args.predictvariance):
            sigma = img[1]
            assert(sigma.shape == (args.resolution, args.resolution))
            sigma_clamped = torch.clamp(sigma, 0.0, 1.0)
            min_val = 0.01
            sigma_log_scaled = (-1.0 / math.log(min_val)) * torch.log(min_val + (1.0 - min_val) * sigma_clamped) + 1.0
            mask = torch.cat((
                torch.zeros((args.resolution, args.resolution, 3)),
                sigma_log_scaled.unsqueeze(-1),
            ), dim=2)
            plt_axis.imshow(mask, interpolation="bicubic")
        plt_axis.axis("off")

    def plot_depthmap(plt_axis, data):
        y = data[0]
        assert(y.shape == (args.resolution,))
        plt_axis.plot(y, c="black")
        if (args.predictvariance):
            sigma = data[1]
            assert(sigma.shape == (args.resolution,))
            plt_axis.plot(y - sigma, c="red")
            plt_axis.plot(y + sigma, c="red")
        plt_axis.set_ylim(-0.5, 1.5)

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

    log_path = "logs/{}_{}".format(args.experiment, timestamp)

    with SummaryWriter(log_path) as writer:

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

                if ((global_iteration + 1) % args.plotinterval) == 0:
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
                    plot_ground_truth(ax_t2, batch_cpu)
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
                    ax_t4.set_yscale('log')

                    # plot the validation loss on a log plot
                    ax_b4.title.set_text("Validation Loss")
                    ax_b4.set_yscale('log')
                    ax_b4.plot(val_loss_x, val_loss_y, c="Red")

                    # Note: calling show or pause will cause a bad time
                    fig.canvas.flush_events()
                    
                    # clear output window and display updated figure
                    # display.clear_output(wait=True)
                    # display.display(plt.gcf())
                    print("Epoch {}, iteration {} of {} ({} %), loss={}".format(e, i, len(train_loader), 100*i//len(train_loader), losses[-1]))

                    if ((global_iteration + 1) % (args.plotinterval * 8)) == 0:
                        plt_screenshot().save(log_path + "/image_" + str(global_iteration + 1) + ".png")

                        # NOTE: this is done in the screenshot branch so that a screenshot with the final
                        # performance is always included
                        if args.iterations is not None and global_iteration > args.iterations:
                            print("Done - desired iteration count was reached")
                            return

                global_iteration += 1

        plt.close('all')

if __name__ == "__main__":
    main()
