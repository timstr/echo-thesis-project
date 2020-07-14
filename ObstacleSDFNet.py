import torch
import torch.nn as nn

from device_dict import DeviceDict

class ObstacleSDFNet(nn.Module):
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
        
        def makeFullyConnected(in_features, out_features, activation=True):
            lin = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=True
            )
            if activation:
                return nn.Sequential(
                    lin,
                    nn.ReLU()
                )
            else:
                return lin
        
        # input size: 4 x 4096
        self.cnv1 = makeConvDown(4, 8) # 8 x 2048
        self.cnv2 = makeConvDown(8, 8) # 8 x 1024
        self.cnv3 = makeConvDown(8, 16) # 16 x 512
        self.cnv4 = makeConvDown(16, 16) # 16 x 256
        self.cnv5 = makeConvDown(16, 32) # 32 x 128
        self.cnv6 = makeConvDown(32, 32) # 32 x 64
        # self.cnv7 = makeConvDown(64, 64) # 64 x 32
        # self.cnv8 = makeConvDown(64, 128) # 128 x 16
        # self.cnv9 = makeConvDown(128, 128) # 128 x 8
        # self.cnv10 = makeConvDown(128, 128) # 128 x 4
        # self.cnv11 = makeConvDown(128, 256) # 256 x 2
        # self.cnv12 = makeConvDown(256, 256) # 256 x 1

        # summary statistics per channel:
        # - first moment (across pixels)
        # - second moment (across pixels)?
        # - average (across channel values)
        # - variance (accross channel values)
        # Total: 4

        # pooling layer: summary statistics of 32 channels across 64 pixels
        # resulting in 32 channels x 4 statistics = 128 features

        # features from x/y field location input: 2
        # total features: 130

        self.fc1 = makeFullyConnected(130, 128)
        self.fc2 = makeFullyConnected(128, 128)
        self.fc3 = makeFullyConnected(128, 128)
        self.fc4 = makeFullyConnected(128, 64)
        self.fc5 = makeFullyConnected(64, 1, activation=False)
        
    def forward(self, d):
        w0  = d['echo_waveshaped']

        B = w0.shape[0]

        w1  = self.cnv1(w0)
        w2  = self.cnv2(w1)
        w3  = self.cnv3(w2)
        w4  = self.cnv4(w3)
        w5  = self.cnv5(w4)
        w6  = self.cnv6(w5)
        # w7  = self.cnv7(w6)
        # w8  = self.cnv8(w7)
        # w9  = self.cnv9(w8)
        # w10 = self.cnv10(w9)
        # w11 = self.cnv11(w10)
        # w12 = self.cnv12(w11)
        # assert(w12.shape == (B, 256, 1))

        wx = w6

        assert(wx.shape == (B, 32, 64))

        ls = torch.linspace(0.0, 1.0, 64).to(wx.device)

        mom1 = torch.sum(
            wx * ls.unsqueeze(0).unsqueeze(0),
            dim=2
        )
        assert(mom1.shape == (B, 32))

        mom2 = torch.sum(
            (wx * (ls.unsqueeze(0).unsqueeze(0) - mom1.unsqueeze(-1)))**2,
            dim=2
        )
        assert(mom2.shape == (B, 32))

        mean = torch.mean(
            wx,
            dim=2
        )
        assert(mean.shape == (B, 32))

        variance = torch.var(
            wx,
            dim=2
        )
        assert(variance.shape == (B, 32))

        summary_stats = torch.stack((
            mom1,
            mom2,
            mean,
            variance
        ), dim=2)
        assert(summary_stats.shape == (B, 32, 4))
        summary_stats = summary_stats.reshape(B, 128)

        yx = d['sdf_location']
        
        assert(yx.shape == (B, 2))

        v = torch.cat(
            (summary_stats, yx),
            dim=1
        )

        assert(v.shape == (B, 130))

        v1 = self.fc1(v)
        v2 = self.fc2(v1)
        v3 = self.fc3(v2)
        v4 = self.fc4(v3)
        v5 = self.fc5(v4)

        assert(v5.shape == (B, 1))

        vfinal = v5

        return DeviceDict({'sdf_value': vfinal})