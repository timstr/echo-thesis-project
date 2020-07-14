import torch
import torch.nn as nn

from device_dict import DeviceDict

class ObstacleMapNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        def makeConvDown1D(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.ReLU()
            )
        
        def make1x1Conv1D(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1
                ),
                nn.ReLU()
            )
        
        def makeConvUp2D(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.ReLU()
            )

        # downsampling convolutions
        # input size: 4 x 4096
        self.convDown1 = makeConvDown1D(4, 8) # 8 x 2048
        self.convDown2 = makeConvDown1D(8, 8) # 8 x 1024
        self.convDown3 = makeConvDown1D(8, 16) # 16 x 512
        self.convDown4 = makeConvDown1D(16, 16) # 16 x 256
        self.convDown5 = makeConvDown1D(16, 32) # 32 x 128
        self.convDown6 = makeConvDown1D(32, 64) # 64 x 64
        self.convDown7 = makeConvDown1D(64, 64) # 64 x 32
        self.convDown8 = makeConvDown1D(64, 64) # 64 x 16
        self.convDown9 = makeConvDown1D(64, 64) # 64 x 8
        self.convDown10 = makeConvDown1D(64, 128) # 128 x 4
        self.convDown11 = makeConvDown1D(128, 128) # 128 x 2
        self.convDown12 = makeConvDown1D(128, 256) # 256 x 1

        # 1x1 convolution (fully-connected) layers
        self.fc1 = make1x1Conv1D(256, 256) # 256 x 1
        self.fc2 = make1x1Conv1D(256, 256) # 256 x 1
        self.fc3 = make1x1Conv1D(256, 256) # 256 x 1
        self.fc4 = make1x1Conv1D(256, 256) # 256 x 1

        # reshaping to 4 x 8 x 8

        self.convUp1 = makeConvUp2D(4, 2) # 2 x 16 x 16
        self.convUp2 = makeConvUp2D(2, 1) # 1 x 32 x 32
        
    def forward(self, d):
        x0 = d['echo_waveshaped']

        B = x0.shape[0]

        xd1 = self.convDown1(x0)
        xd2 = self.convDown2(xd1)
        xd3 = self.convDown3(xd2)
        xd4 = self.convDown4(xd3)
        xd5 = self.convDown5(xd4)
        xd6 = self.convDown6(xd5)
        xd7 = self.convDown7(xd6)
        xd8 = self.convDown8(xd7)
        xd9 = self.convDown9(xd8)
        xd10 = self.convDown10(xd9)
        xd11 = self.convDown11(xd10)
        xd12 = self.convDown12(xd11)
        
        xi1 = self.fc1(xd12)
        xi2 = self.fc2(xi1)
        xi3 = self.fc3(xi2)
        xi4 = self.fc4(xi3)
        
        xrs = xi4.reshape(B, 4, 8, 8)

        xd1 = self.convUp1(xrs)
        xd2 = self.convUp2(xd1)
        
        xfinal = xd2.squeeze(1)

        return DeviceDict({'obstacles': xfinal})