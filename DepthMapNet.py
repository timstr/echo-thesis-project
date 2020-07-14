import torch
import torch.nn as nn

from device_dict import DeviceDict

class DepthMapNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        def makeConvDown(in_channels, out_channels):
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
        
        def makeConvUp(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.ReLU()
            )

        # TODO: try concatenating instead of adding
        
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


        
        self.convUp1 = makeConvUp(256, 128)
        self.convUp2 = makeConvUp(128, 64)
        self.convUp3 = makeConvUp(64, 32)
        self.convUp4 = makeConvUp(32, 16)
        self.convUp5 = makeConvUp(16, 8)
        self.convUp6 = makeConvUp(8, 4)
        self.convUp7 = makeConvUp(4, 2)
        
        self.finalConv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1)
        
    def forward(self, d):
        x0 = d['echo_waveshaped']
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
        
        
        xd1 = self.convUp1(xu12)
        xd2 = self.convUp2(xd1)
        xd3 = self.convUp3(xd2)
        xd4 = self.convUp4(xd3)
        xd5 = self.convUp5(xd4)
        xd6 = self.convUp6(xd5)
        xd7 = self.convUp7(xd6)
        
        xfinal = self.finalConv(xd7).squeeze(1) * 300

        return DeviceDict({'depthmap': xfinal})