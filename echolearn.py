import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math

from device_dict import DeviceDict
from dataset import WaveSimDataset
from unet_cnn import UNetCNN

def main():
    ecds = WaveSimDataset(data_folder="dataset/v5")

    batch_size = 16

    val_ratio = 0.1
    val_size = int(len(ecds)*val_ratio)
    indices_val = list(range(0, val_size))
    indices_train = list(range(val_size, len(ecds)))

    val_set   = torch.utils.data.Subset(ecds, indices_val)
    train_set = torch.utils.data.Subset(ecds, indices_train)

    # define the dataset loader (batch size, shuffling, ...)
    collate_fn_device = lambda batch : DeviceDict(torch.utils.data.dataloader.default_collate(batch)) # collate_fn_device is necessary to preserve our custom dictionary during the collection of samples fetched from the dataset into a Tensor batch. 
    # Hopefully, one day, pytorch might change the default collate to pretain the mapping type. Currently all Mapping objects are converted to dict. Anyone wants to create a pull request? Would need to be changed in 
    # pytorch/torch/utils/data/_utils/collate.py :     elif isinstance(data, container_abcs.Mapping): return {key: default_convert(data[key]) for key in data}
    # pytorch/torch/utils/data/_utils/pin_memory.py : if isinstance(data, container_abcs.Mapping): return {k: pin_memory(sample) for k, sample i                        n data.items()}
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
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn_device
    )

    loss_function = torch.nn.functional.mse_loss

    ##############################################
    # Helper function to compute validation loss #
    ##############################################
    def compute_averaged_loss(model, loader):
        it = iter(loader)
        losses = []
        for i in range(min(len(loader), 100)): # range(len(val_iter)):
            batch = next(it).to('cuda')
            pred = model(batch)
            depthmap_pred = pred['heatmap']
            depthmap_gt = batch['heatmap']
            loss = loss_function(depthmap_pred, depthmap_gt)
            losses.append(loss.item())
        return np.mean(np.asarray(losses))
        
    def training_loss(model):
        return compute_averaged_loss(model, train_loader)
        
    def validation_loss(model):
        return compute_averaged_loss(model, val_loader)

    network = UNetCNN().cuda()

    def save_network(filename):
            print("Saving model to \"{}\"".format(filename))
            torch.save(network.state_dict(), filename)

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    plt.ion()
    # fig = plt.figure(figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')
    # axes=fig.subplots(2,3)

    # Oh boy do I hate this interface
    ax_tl = plt.subplot(231)
    ax_tm = plt.subplot(232)
    ax_tr = plt.subplot(233)
    ax_bl = plt.subplot(234, projection="polar")
    ax_bm = plt.subplot(235)
    ax_br = plt.subplot(236)

    angles = np.linspace(0.0, math.pi * 2.0, 128)
    

    num_epochs = 1000
    losses = []
    val_loss_y = []
    val_loss_x = []
    best_val_loss = np.inf
    key = 'heatmap'
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

            if i%25==0:
                curr_val_loss = validation_loss(network)
                if curr_val_loss < best_val_loss:
                    best_val_loss = curr_val_loss
                    save_network("echolearn_model_best.dat")


                val_loss_x.append(len(losses))
                val_loss_y.append(curr_val_loss)
                
                # clear figures for a new update
                ax_tl.cla()
                ax_tm.cla()
                ax_tr.cla()
                ax_bl.cla()
                ax_bm.cla()
                ax_br.cla()
                    
                # plot the ground truth heat map
                heatmap_gt = batch_cpu['heatmap'][0].detach()
                ax_tl.plot(heatmap_gt)
                ax_tl.set_ylim(0, 1)
                # plot the predicted heat map
                heatmap_pred = pred_cpu['heatmap'][0].detach()
                # axes[0,1].plot(smooth(heatmap_pred, 255))
                ax_tm.plot(heatmap_pred)
                ax_tm.set_ylim(0, 1)
                
                # plot the training and validation errors on a log plot
                ax_tr.scatter(range(len(losses)), losses, s=1.0)
                ax_tr.set_yscale('log')
                ax_tr.set_ylim(1e-4, 1)
                ax_tr.plot(val_loss_x, val_loss_y, c="Red")
                
                # plot the depthmap
                ax_bl.set_ylim(0, math.hypot(256, 256))
                ax_bl.plot(angles, batch_cpu['depthmap'][0].detach())

                # plot the input waveforms
                for j in range(batch_cpu['echo_waveshaped'].shape[1]):
                    ax_bm.plot(batch_cpu['echo_waveshaped'][0][j].detach())
                ax_bm.set_ylim(-1, 1)

                plt.show()
                plt.pause(0.001)
                
                # clear output window and display updated figure
                # display.clear_output(wait=True)
                # display.display(plt.gcf())
                print("Epoch {}, iteration {} of {} ({} %), loss={}".format(e, i, len(train_loader), 100*i//len(train_loader), losses[-1]))

        if (e + 1) % 5 == 0:
            save_network("echolearn_model_most_recent.dat")

    plt.close('all')

if __name__ == "__main__":
    main()
