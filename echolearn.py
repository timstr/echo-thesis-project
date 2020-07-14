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
from featurize import shortest_distance_to_obstacles, make_sdf_image_gt, make_sdf_image_pred
from custom_collate_fn import custom_collate

# from unet_cnn import UNetCNN
# from DepthMapNet import DepthMapNet
# from ObstacleMapNet import ObstacleMapNet
from ObstacleSDFNet import ObstacleSDFNet

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
    parser.add_argument("--showloss", type=bool, dest="showloss", required=True)
    parser.add_argument("--dataset", type=str, dest="dataset", required=True)
    args = parser.parse_args()

    ecds = WaveSimDataset(
        data_folder="dataset/{}".format(args.dataset),
        permute=False,
        samples_per_example=1
    )

    batch_size = 64

    val_ratio = 0.1
    val_size = int(len(ecds)*val_ratio)
    indices_val = list(range(0, val_size))
    indices_train = list(range(val_size, len(ecds)))

    val_set   = torch.utils.data.Subset(ecds, indices_val)
    train_set = torch.utils.data.Subset(ecds, indices_train)

    # define the dataset loader (batch size, shuffling, ...)
    collate_fn_device = lambda batch : DeviceDict(custom_collate(batch))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False, # Note, setting pin_memory=False to avoid the pin_memory call
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_device
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_device
    )

    loss_function = torch.nn.functional.mse_loss

    ##############################################
    # Helper function to compute validation loss #
    ##############################################
    def compute_averaged_loss(model, loader, key):
        it = iter(loader)
        losses = []
        for i in range(min(len(loader), 100)): # range(len(val_iter)):
            batch = next(it).to('cuda')
            pred = model(batch)
            depthmap_pred = pred[key]
            depthmap_gt = batch[key]
            loss = loss_function(depthmap_pred, depthmap_gt)
            losses.append(loss.item())
        return np.mean(np.asarray(losses))
        
    def training_loss(model, key):
        return compute_averaged_loss(model, train_loader, key)
        
    def validation_loss(model, key):
        return compute_averaged_loss(model, val_loader, key)

    # network = UNetCNN().cuda()
    # network = DepthMapNet().cuda()
    # network = ObstacleMapNet().cuda()
    network = ObstacleSDFNet().cuda()

    def save_network(filename):
        filename = "models/" + filename
        print("Saving model to \"{}\"".format(filename))
        torch.save(network.state_dict(), filename)

    def restore_network(filename):
        filename = "models/" + filename
        print("Loading model from \"{}\"".format(filename))
        network.load_state_dict(torch.load(filename))
        network.eval()

    # restore_network("time_representation_test_even_less_convs_16_permuted_examples_02-07-2020_09-02-04_model_best.dat")

    # def render_test_animation():
    #     ds = WaveSimDataset(data_folder="dataset/v7_test", permute=False, samples_per_example=1)
    #     ld = torch.utils.data.DataLoader(
    #         ds,
    #         batch_size=1,
    #         num_workers=0,
    #         pin_memory=False, # Note, setting pin_memory=False to avoid the pin_memory call
    #         shuffle=False,
    #         drop_last=True,
    #         collate_fn=collate_fn_device
    #     )
    #     plt.axis('off')
    #     # fig = plt.figure(
    #     #     figsize=(8,8),
    #     #     dpi=64
    #     # )
    #     it = iter(ld)
    #     def next_data():
    #         batch = next(it).to('cuda')
    #         return make_sdf_image_pred(batch, 32, network)
    #         # return make_sdf_image_gt(batch, 32)
        
    #     num_frames = len(ld)
    #     def animate(i):
    #         plt.clf()
    #         plt.imshow(next_data(), vmin=0.0, vmax=0.5, cmap='hsv')
    #         progress_bar(i, num_frames)
    #     output_path = "test set sdf prediction.mp4"
    #     fps = 30
    #     ani = matplotlib.animation.FuncAnimation(
    #         plt.gcf(),
    #         animate,
    #         frames=num_frames,
    #         interval = 1000/fps
    #     )

    #     ani.save(output_path)

    #     sys.stdout.write("\nSaved animation to {}".format(output_path))

    # render_test_animation()
    # return

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    log_path = "logs/{}_{}".format(args.experiment, timestamp)

    with SummaryWriter(log_path) as writer:

        plt.ion()
        # fig = plt.figure(figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')
        # axes=fig.subplots(2,3)

        # Oh boy do I hate this interface
        # ax_t1 = plt.subplot(241)
        # ax_t2 = plt.subplot(242)
        # ax_t3 = plt.subplot(243)
        # ax_t4 = plt.subplot(244)#, projection="polar")
        # ax_b1 = plt.subplot(245)
        # ax_b2 = plt.subplot(246)
        # ax_b3 = plt.subplot(247)
        # ax_b4 = plt.subplot(248)

        numcols = 4 if args.showloss else 3
        fig, axes = plt.subplots(2, 4, figsize=(16, 6), dpi=100)

        ax_t1 = axes[0,0]
        ax_t2 = axes[0,1]
        ax_t3 = axes[0,2]
        ax_b1 = axes[1,0]
        ax_b2 = axes[1,1]
        ax_b3 = axes[1,2]

        if args.showloss:
            ax_t4 = axes[0,3]
            ax_b4 = axes[1,3]

        # angles = np.linspace(0.0, math.pi * 2.0, 128)
        
        sdf_img_size = 32

        visualization_interval = len(train_loader) * 4

        num_epochs = 1000000
        losses = []
        val_loss_y = []
        val_loss_x = []
        best_val_loss = np.inf
        key = 'sdf_value'
        global_iteration = 0
        for e in range(num_epochs):
            train_iter = iter(train_loader)
            for i in range(len(train_loader)):
                batch_cpu = next(train_iter)
                batch_gpu = batch_cpu.to('cuda')
                
                pred_gpu = network(batch_gpu)
                pred_cpu = pred_gpu.to('cpu')
                
                loss = loss_function(pred_gpu[key], batch_gpu[key])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                writer.add_scalar("training loss", loss.item(), global_iteration)
                
                progress_bar((global_iteration) % visualization_interval, visualization_interval)

                if ((global_iteration + 1) % visualization_interval) == 0:
                    print("")
                    val_batch_cpu = next(iter(val_loader))
                    val_batch_gpu = val_batch_cpu.to('cuda')

                    curr_val_loss = validation_loss(network, key)
                    if curr_val_loss < best_val_loss:
                        best_val_loss = curr_val_loss
                        save_network("{}_{}_model_best.dat".format(args.experiment, timestamp))


                    val_loss_x.append(len(losses))
                    val_loss_y.append(curr_val_loss)
                    
                    writer.add_scalar("validation loss", curr_val_loss, global_iteration)

                    # clear figures for a new update
                    ax_t1.cla()
                    ax_b1.cla()
                    ax_t2.cla()
                    ax_b2.cla()
                    ax_t3.cla()
                    ax_b3.cla()
                    if args.showloss:
                        ax_t4.cla()
                        ax_b4.cla()
                        
                    # plot the input waveforms
                    ax_t1.title.set_text("Waveform (raw)")
                    echo_raw = batch_cpu['echo_raw'][0].detach()
                    for j in range(echo_raw.shape[0]):
                        ax_t1.plot(echo_raw[j].detach())
                    ax_t1.set_ylim(-1, 1)
                    
                    ax_b1.title.set_text("Waveform (waveshaped)")
                    echo_waveshaped = batch_cpu['echo_waveshaped'][0].detach()
                    for j in range(echo_waveshaped.shape[0]):
                        ax_b1.plot(echo_waveshaped[j].detach())
                    ax_b1.set_ylim(-1, 1)
                    
                    
                    # plot the ground truth obstacles (training set)
                    ax_t2.title.set_text("Ground Truth SDF (train)")
                    ax_t2.imshow(make_sdf_image_gt(batch_cpu, sdf_img_size), vmin=0, vmax=0.5, cmap='hsv')
                    
                    # plot the predicted sdf (training set)
                    ax_b2.title.set_text("Predicted SDF (train)")
                    ax_b2.imshow(make_sdf_image_pred(batch_gpu, sdf_img_size, network), vmin=0, vmax=0.5, cmap='hsv')
                    
                    # plot the ground truth obstacles (validation set)
                    ax_t3.title.set_text("Ground Truth SDF (validation)")
                    ax_t3.imshow(make_sdf_image_gt(val_batch_cpu, sdf_img_size), vmin=0, vmax=0.5, cmap='hsv')
                    
                    # plot the predicted obstacles (validation set)
                    ax_b3.title.set_text("Predicted SDF (validation)")
                    ax_b3.imshow(make_sdf_image_pred(val_batch_gpu, sdf_img_size, network), vmin=0, vmax=0.5, cmap='hsv')
                    
                    if args.showloss:
                        # plot the training and validation errors on a log plot
                        ax_t4.title.set_text("Loss")
                        ax_t4.scatter(range(len(losses)), losses, s=1.0)
                        ax_t4.set_yscale('log')
                        ax_t4.plot(val_loss_x, val_loss_y, c="Red")

                    # Note: calling show or pause will cause a bad time
                    # plt.show()
                    # plt.pause(0.001)
                    plt.gcf().canvas.flush_events()
                    
                    # clear output window and display updated figure
                    # display.clear_output(wait=True)
                    # display.display(plt.gcf())
                    print("Epoch {}, iteration {} of {} ({} %), loss={}".format(e, i, len(train_loader), 100*i//len(train_loader), losses[-1]))

                    # if (global_iteration % 1000 == 0):
                    plt_screenshot().save(log_path + "/image_" + str(global_iteration + 1) + ".jpg")

                global_iteration += 1
            if (e + 1) % 5 == 0:
                save_network("{}_{}_model_latest.dat".format(args.experiment, timestamp))

        plt.close('all')

if __name__ == "__main__":
    main()
