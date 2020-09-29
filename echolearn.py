import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import math
import datetime
import PIL.Image

from device_dict import DeviceDict
from dataset import WaveSimDataset
from progress_bar import progress_bar
from featurize import make_sdf_image_gt, make_sdf_image_pred, make_receiver_indices, make_heatmap_image_gt, make_heatmap_image_pred, make_depthmap_gt, make_depthmap_pred
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
    parser.add_argument("--dataset", type=str, dest="dataset", required=True)
    parser.add_argument("--numexamples", type=int, dest="numexamples", default=None)
    parser.add_argument("--batchsize", type=int, dest="batchsize", default=64)
    parser.add_argument("--receivers", type=int, choices=[1, 2, 4, 8, 16, 32, 64], dest="receivers", required=True)
    parser.add_argument("--arrangement", type=str, choices=["flat", "grid"], dest="arrangement", required=True)
    parser.add_argument("--maxobstacles", type=int, dest="maxobstacles", default=None)
    parser.add_argument("--circlesonly", dest="circlesonly", default=False, action="store_true")
    parser.add_argument("--nosave", dest="nosave", default=False, action="store_true")
    parser.add_argument("--iterations", type=int, dest="iterations", default=None)
    parser.add_argument("--implicitfunction", dest="implicitfunction", default=False, action="store_true")
    parser.add_argument("--gradientregularizer", dest="gradientregularizer", type=float, required=False)
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

    output_resolution = 32 # TODO: allow this to be customized? This has some implications for the internal bandwidth of the network

    ecds = WaveSimDataset(
        data_folder="dataset/{}".format(args.dataset),
        samples_per_example=128,
        num_examples=args.numexamples,
        max_obstacles=args.maxobstacles,
        receiver_indices=receiver_indices,
        circles_only=args.circlesonly,
        input_representation=args.nninput,
        output_representation=args.nnoutput,
        implicit_function=args.implicitfunction
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

    # Encourages one of two things:
    # - Either the value is zero and the gradient is zero, or
    # - the value is nonzero and the gradient's magnitude is 1
    # Parameters:
    # - value: value of the function
    # - gradient: magnitude of the derivative of the value w.r.t. its spatial coordinates
    # - kZeroTolerance: varies the radius of the region around zero that is considered to be zero
    def zeroZeroXOneLoss(value, gradientMag):
        return torch.mean(torch.sqrt((value**2 + gradientMag**2) * (gradientMag - 1.0)**2))
    
    if (args.gradientregularizer is not None):
        assert(args.nnoutput == "sdf")
        if (args.implicitfunction):
            def gradientRegularizedLoss(batch_input, batch_output):
                x1 = batch_input["output"]
                x2 = batch_output["output"]
                params = batch_gpu["params"]
                mse = torch.nn.functional.mse_loss(x1, x2)
                grad, = torch.autograd.grad(mse, params, retain_graph=True)
                grad_mag = torch.sqrt(torch.sum(grad**2, dim=2))
                grad_penality = zeroZeroXOneLoss(x2, grad_mag)
                return mse + grad_penality * args.gradientregularizer
            loss_function = gradientRegularizedLoss
        else:
            def gradientRegularizedLoss(batch_input, batch_output):
                x1 = batch_input["output"]
                x2 = batch_output["output"]
                mse = torch.nn.functional.mse_loss(x1, x2)
                # NOTE: assuming the square output region has unit size, each
                # pixel has size (1.0 / output_resolution). The following two
                # lines are a finite-differences estimate of the image gradient,
                # with sampling points chosen at a distance of two pixels apart
                # (for symmetry), or (2.0 / output_resolution). The difference
                # needs to be divided by the distance, which is equivalent
                # to multiplying by (output_resolution / 2.0)
                dy = (x2[:, 2:, 1:-1] - x2[:, :-2, 1:-1]) * (output_resolution / 2.0)
                dx = (x2[:, 1:-1, 2:] - x2[:, 1:-1, :-2]) * (output_resolution / 2.0)
                grad_mag = torch.sqrt(dy**2 + dx**2)
                value = x2[:, 1:-1, 1:-1]
                grad_penality = zeroZeroXOneLoss(value, grad_mag)
                return mse + grad_penality * args.gradientregularizer
            loss_function = gradientRegularizedLoss
    else:
        def bogStandardLoss(batch_input, batch_output):
            x1 = batch_input["output"]
            x2 = batch_output["output"]
            return torch.nn.functional.mse_loss(x1, x2)
        loss_function = bogStandardLoss
        

    ##############################################
    # Helper function to compute validation loss #
    ##############################################
    def compute_averaged_loss(model, loader):
        print("Computing validation loss (MSE)...")
        it = iter(loader)
        losses = []
        for i in range(len(loader)):
            batch = next(it).to('cuda')
            pred = model(batch)
            depthmap_pred = pred["output"]
            depthmap_gt = batch["output"]
            loss = torch.nn.functional.mse_loss(depthmap_pred, depthmap_gt)
            losses.append(loss.item())
        print("Computing validation loss (MSE)... Done.")
        return np.mean(np.asarray(losses))
        
    def training_loss(model):
        return compute_averaged_loss(model, train_loader)
        
    def validation_loss(model):
        x = compute_averaged_loss(model, val_loader)
        return x

    NetworkType = SimpleNN if args.simplenn else EchoLearnNN

    network = NetworkType(
        num_input_channels=args.receivers,
        num_implicit_params=implicit_params,
        input_format=input_format,
        output_format=output_format,
        output_resolution=(None if args.implicitfunction else output_resolution)
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

    def plot_inputs(pltx_axis, batch):
        the_input = batch['input'][0].detach()
        if (len(the_input.shape) == 2):
            for j in range(args.receivers):
                pltx_axis.plot(the_input[j].detach())
            pltx_axis.set_ylim(-1, 1)
        else:
            the_input_min = torch.min(the_input)
            the_input_max = torch.max(the_input)
            the_input = (the_input - the_input_min) / (the_input_max - the_input_min)
            spectrogram_img_grid = torchvision.utils.make_grid(
                the_input.unsqueeze(1).repeat(1, 3, 1, 1),
                nrow=1
            )
            pltx_axis.imshow(spectrogram_img_grid.permute(1, 2, 0))

    def plot_ground_truth(plt_axis, batch):
        if args.nnoutput == "sdf":
            img = make_sdf_image_gt(batch, output_resolution)
            plt_axis.imshow(img, vmin=0, vmax=0.5, cmap='hsv')
        elif args.nnoutput == "heatmap":
            img = make_heatmap_image_gt(batch, output_resolution)
            plt_axis.imshow(img, vmin=0, vmax=1.0)
        elif args.nnoutput == "depthmap":
            arr = make_depthmap_gt(batch, output_resolution)
            plt_axis.plot(arr)
            plt_axis.set_ylim(0.0, 1.0)
        else:
            raise Exception("Unrecognized output representation")

    def plot_prediction(plt_axis, batch, network):
        if args.implicitfunction:
            if args.nnoutput == "sdf":
                img = make_sdf_image_pred(batch, output_resolution, network)
                plt_axis.imshow(img, vmin=0, vmax=0.5, cmap='hsv')
            elif args.nnoutput == "heatmap":
                img = make_heatmap_image_pred(batch, output_resolution, network)
                plt_axis.imshow(img, vmin=0, vmax=1.0)
            elif args.nnoutput == "depthmap":
                arr = make_depthmap_pred(batch, output_resolution, network)
                plt_axis.plot(arr)
                plt_axis.set_ylim(-0.1, 1.1)
            else:
                raise Exception("Unrecognized output representation")
        else:
            # non-implicit function
            output = network(batch)['output'][0].detach().cpu().numpy()
            if args.nnoutput == "sdf":
                assert(output.shape == (output_resolution, output_resolution))
                plt_axis.imshow(output, vmin=0, vmax=0.5, cmap='hsv')
            elif args.nnoutput == "heatmap":
                assert(output.shape == (output_resolution, output_resolution))
                plt_axis.imshow(output, vmin=0, vmax=1.0)
            elif args.nnoutput == "depthmap":
                assert(len(output.shape) == 1)
                assert(output.shape[0] == output_resolution)
                plt_axis.plot(output)
                plt_axis.set_ylim(-0.1, 1.1)
            else:
                raise Exception("Unrecognized output representation")

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    log_path = "logs/{}_{}".format(args.experiment, timestamp)

    with SummaryWriter(log_path) as writer:

        plt.ion()

        fig, axes = plt.subplots(2, 4, figsize=(16, 6), dpi=100)

        ax_t1 = axes[0,0]
        ax_t2 = axes[0,1]
        ax_t3 = axes[0,2]
        ax_b1 = axes[1,0]
        ax_b2 = axes[1,1]
        ax_b3 = axes[1,2]
        ax_t4 = axes[0,3]
        ax_b4 = axes[1,3]

        if (args.nosave):
            print("NOTE: networks are not being saved")

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
                
                loss = loss_function(pred_gpu, batch_gpu)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                writer.add_scalar("training loss", loss.item(), global_iteration)
                
                progress_bar((global_iteration) % args.plotinterval, args.plotinterval)

                if ((global_iteration + 1) % args.validationinterval) == 0:
                    curr_val_loss = validation_loss(network)
                    if curr_val_loss < best_val_loss:
                        best_val_loss = curr_val_loss
                        if not args.nosave:
                            save_network("{}_{}_model_best.dat".format(args.experiment, timestamp))


                    val_loss_x.append(len(losses))
                    val_loss_y.append(curr_val_loss)
                    
                    writer.add_scalar("validation loss", curr_val_loss, global_iteration)

                if ((global_iteration + 1) % args.plotinterval) == 0:
                    print("")
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

                    plt.gcf().suptitle(args.experiment)
                        
                    # plot the input waveforms
                    ax_t1.title.set_text("Input (train)")
                    plot_inputs(ax_t1, batch_cpu)
                    ax_b1.title.set_text("Input (val.)")
                    plot_inputs(ax_b1, val_batch_cpu)
                    
                    # plot the ground truth obstacles
                    ax_t2.title.set_text("Ground Truth (train)")
                    plot_ground_truth(ax_t2, batch_cpu)
                    ax_b2.title.set_text("Ground Truth (val.)")
                    plot_ground_truth(ax_b2, val_batch_cpu)
                    
                    # plot the predicted sdf
                    ax_t3.title.set_text("Prediction (train)")
                    plot_prediction(ax_t3, batch_gpu, network)
                    ax_b3.title.set_text("Prediction (val.)")
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
                    # plt.show()
                    # plt.pause(0.001)
                    plt.gcf().canvas.flush_events()
                    
                    # clear output window and display updated figure
                    # display.clear_output(wait=True)
                    # display.display(plt.gcf())
                    print("Epoch {}, iteration {} of {} ({} %), loss={}".format(e, i, len(train_loader), 100*i//len(train_loader), losses[-1]))

                    if ((global_iteration + 1) % (args.plotinterval * 8)) == 0:
                        plt_screenshot().save(log_path + "/image_" + str(global_iteration + 1) + ".jpg")

                        # NOTE: this is done in the screenshot branch so that a screenshot with the final
                        # performance is always included
                        if args.iterations is not None and global_iteration > args.iterations:
                            print("Done - desired iteration count was reached")
                            return

                global_iteration += 1
            if (e + 1) % 5 == 0:
                if not args.nosave:
                    save_network("{}_{}_model_latest.dat".format(args.experiment, timestamp))


        plt.close('all')

if __name__ == "__main__":
    main()
