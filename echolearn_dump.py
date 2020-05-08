import math
import random
import time
import pickle
import os.path
import glob
from IPython.display import clear_output

import torch
import torch.nn as nn

import numpy as np

# %matplotlib notebook
import matplotlib.pyplot as plt
import matplotlib.animation

from IPython.display import HTML

import PIL

import scipy.io.wavfile as wf

# signed, clipped logarithm
def sclog(t):
    max_val = 1e0
    min_val = 1e-4
    signs = torch.sign(t)
    t = torch.abs(t)
    t = torch.clamp(t, min=min_val, max=max_val)
    t = torch.log(t)
    t = (t - math.log(min_val)) / (math.log(max_val) - math.log(min_val))
    t = t * signs
    return t

def get_dataset_item(i):
    path = "dataset/v3"
    fname = "{0}/example {1}.pkl".format(path, i)
    assert(os.path.isfile(fname))
    with open(fname, "rb") as file:
        data = pickle.load(file)
    return data

def make_first_echo_bump(mindist, length):
    width = 0.01
    mindist = mindist / 1050.0
    x0 = torch.linspace(0.0, 1.0, steps=length)
    x1 = torch.exp(-((x0 - mindist) / width)**2)
    return x1

barrier, echo, depthmap = get_dataset_item(2)
plt.imshow(barrier)
plt.show()
plt.plot(echo)
plt.show()
plt.plot(sclog(torch.tensor(echo)))
plt.show()
plt.plot(make_first_echo_bump(np.min(depthmap), len(echo)))
plt.show()
plt.plot(depthmap)
plt.show()

# utility dictionary that can move tensor values between devices via the 'to(device)' function
from collections import OrderedDict 
class DeviceDict(dict):
    # following https://stackoverflow.com/questions/3387691/how-to-perfectly-override-a-dict
    def __init__(self, *args):
        super(DeviceDict, self).__init__(*args)
    def to(self, device):
        dd = DeviceDict() # return shallow copy
        for k,v in self.items():
            if torch.is_tensor(v):
                dd[k] = v.to(device)
            else:
                dd[k] = v
        return dd

class WaveSimDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder):
        super(WaveSimDataset).__init__()
        self.data = []
        print("Loading data into memory...")
        for path in glob.glob("{}/example *.pkl".format(data_folder)):
            with open(path, "rb") as file:
                _, echo, depthmap = pickle.load(file)
            self.data.append((echo, depthmap))
        print("Done.")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        echo, depthmap = self.data[idx]
        echo = sclog(torch.tensor(echo))
        echo = torch.tensor(echo).permute(1, 0).float().detach()
        depthmap = torch.tensor(depthmap).float().detach()
        assert(echo.shape == (4, 8192))
        assert(depthmap.shape == (128,))
        mindist = torch.min(depthmap).float().detach()
        heatmap = make_first_echo_bump(mindist, 8192).float().detach()
        return DeviceDict({
            'echo': echo,
            'depthmap': depthmap,
            'mindist': mindist,
            'heatmap': heatmap
        });

# setting up the dataset and train/val splits
ecds = WaveSimDataset(data_folder="dataset/v3")

val_ratio = 0.2
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
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 2, num_workers=0, pin_memory=False, shuffle=True, drop_last=True, collate_fn=collate_fn_device) # Note, setting pin_memory=False to avoid the pin_memory call
val_loader = torch.utils.data.DataLoader(val_set, batch_size = 2, num_workers=0, pin_memory=False, shuffle=False, drop_last=True, collate_fn=collate_fn_device)

loss_function = torch.nn.functional.mse_loss

##############################################
# Helper function to compute validation loss #
##############################################
def compute_averaged_loss(model, loader):
    it = iter(loader)
    losses = []
    for i in range(100): # range(len(val_iter)):
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
    

def smooth(x, window_size):
    assert(window_size > 0)
    assert((window_size % 2) == 1)
    smooth = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=window_size, padding=((window_size - 1) // 2), bias=False)
    kernel = torch.ones(window_size) / window_size
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    smooth.weight = nn.Parameter(kernel, requires_grad=False)
    return smooth(x.unsqueeze(0)).squeeze(0).detach()

class MyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        def makeConvDown(in_channels, out_channels):
            return nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )
        
        def makeConvUp(in_channels, out_channels):
            return nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            )
        
        self.convDown1 = makeConvDown(4, 8)
        self.convDown2 = makeConvDown(8, 8)
        self.convDown3 = makeConvDown(8, 16)
        self.convDown4 = makeConvDown(16, 16)
        self.convDown5 = makeConvDown(16, 32)
        self.convDown6 = makeConvDown(32, 64)
        self.convDown7 = makeConvDown(64, 64)
        self.convDown8 = makeConvDown(64, 64)
        self.convDown9 = makeConvDown(64, 64)
        self.convDown10 = makeConvDown(64, 128)
        self.convDown11 = makeConvDown(128, 128)
        self.convDown12 = makeConvDown(128, 256)
        self.convDown13 = makeConvDown(256, 256)
        
        self.convUp13 = makeConvUp(256, 256)
        self.convUp12 = makeConvUp(256, 128)
        self.convUp11 = makeConvUp(128, 128)
        self.convUp10 = makeConvUp(128, 64)
        self.convUp9 = makeConvUp(64, 64)
        self.convUp8 = makeConvUp(64, 64)
        self.convUp7 = makeConvUp(64, 64)
        self.convUp6 = makeConvUp(64, 32)
        self.convUp5 = makeConvUp(32, 16)
        self.convUp4 = makeConvUp(16, 16)
        self.convUp3 = makeConvUp(16, 8)
        self.convUp2 = makeConvUp(8, 8)
        self.convUp1 = makeConvUp(8, 4)
        
        self.finalConv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, stride=1)
        
    def forward(self, d):
        x0 = d['echo']
        xu1 = self.convDown1(x0)
        xu2 = self.convDown2(xu1)
        xu3 = self.convDown3(xu2)
        xu4 = self.convDown4(xu3)
        xu5 = self.convDown5(xu4)
        xu6 = self.convDown6(xu5)
        xu7 = self.convDown7(xu6)
        xu8 = self.convDown8(xu7)
        xu9 = self.convDown9(xu8)
        xu10 = self.convDown10(xu9)
        xu11 = self.convDown11(xu10)
        xu12 = self.convDown12(xu11)
        xu13 = self.convDown13(xu12)
        
        
        xd13 = self.convUp13(xu13) + xu12
        xd12 = self.convUp12(xd13) + xu11
        xd11 = self.convUp11(xd12) + xu10
        xd10 = self.convUp10(xd11) + xu9
        xd9 = self.convUp9(xd10) + xu8
        xd8 = self.convUp8(xd9) + xu7
        xd7 = self.convUp7(xd8) + xu6
        xd6 = self.convUp6(xd7) + xu5
        xd5 = self.convUp5(xd6) + xu4
        xd4 = self.convUp4(xd5) + xu3
        xd3 = self.convUp3(xd4) + xu2
        xd2 = self.convUp2(xd3) + xu1
        xd1 = self.convUp1(xd2) + x0
        
        xfinal = self.finalConv(xd1).squeeze(1)
        
        return DeviceDict({'heatmap': xfinal})
        
network = MyNetwork().cuda()

%matplotlib inline
from IPython import display

optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

fig=plt.figure(figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')
axes=fig.subplots(2,3)

num_epochs = 1000
losses = []
val_loss_y = []
val_loss_x = []
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
            val_loss_x.append(len(losses))
            val_loss_y.append(validation_loss(network))
            
            # clear figures for a new update
            for ax in axes:
                for a in ax:
                    a.cla()
                
            # plot the ground truth heat map
            heatmap_gt = batch_cpu['heatmap'][0].detach()
            axes[0,0].plot(heatmap_gt)
            axes[0,0].set_ylim(0, 1)
            # plot the predicted heat map
            heatmap_pred = pred_cpu['heatmap'][0].detach()
            # axes[0,1].plot(smooth(heatmap_pred, 255))
            axes[0,1].plot(heatmap_pred)
            axes[0,1].set_ylim(0, 1)
            
            # plot the training error on a log plot
            axes[0,2].scatter(range(len(losses)), losses, s=1.0)
            axes[0,2].set_yscale('log')
            axes[0,2].plot(val_loss_x, val_loss_y, c="Red")
            
            # plot the input waveforms
            for i in range(batch_cpu['echo'].shape[1]):
                axes[1,0].plot(batch_cpu['echo'][0][i].detach())
                axes[1,1].plot(batch_cpu['echo'][0][i].detach())
            axes[1,0].set_ylim(-1, 1)
            axes[1,1].set_ylim(-1, 1)
            
            # clear output window and display updated figure
            display.clear_output(wait=True)
            display.display(plt.gcf())
            print("Epoch {}, iteration {} of {} ({} %), loss={}".format(e, i, len(train_loader), 100*i//len(train_loader), losses[-1]))
plt.close('all')

plt.scatter(range(len(losses)), losses, s=1.0)
plt.show()

print("Training loss: ", training_loss(network))
print("Valdiation loss: ", validation_loss(network))