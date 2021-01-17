from device_dict import DeviceDict
from reshape_layer import Reshape
import torch
import torch.nn as nn

def circle_c(*args):
    return torch.cat(args, dim=1)

def slash4(x=None):
    if x is None:
        return nn.MaxPool2d(4, 4)
    else:
        return nn.functional.max_pool2d(x, 4, 4)

def defconv3x3_x2(in_features, middle_features, out_features):
    # NOTE: different padding modes and strides do not appear to be available for DeformConv2D
    return nn.Sequential(
        # DeformConv2D(in_features, middle_features, kernel_size=3, padding=1),
        nn.Conv2d(in_features, middle_features, kernel_size=3, stride=1, padding=1, padding_mode="same"),
        # nn.BatchNorm2d(middle_features),
        nn.ReLU(),
        nn.Conv2d(middle_features, out_features, kernel_size=3, stride=1, padding=1, padding_mode="same"),
        # DeformConv2D(middle_features, out_features, kernel_size=3, padding=1),
        # nn.BatchNorm2d(out_features),
        nn.ReLU()
    )

def conv3x3_x2(in_features, middle_features, out_features):
    return nn.Sequential(
        nn.Conv2d(in_features, middle_features, kernel_size=3, padding=1, padding_mode="same", stride=1),
        # nn.BatchNorm2d(middle_features),
        nn.ReLU(),
        nn.Conv2d(middle_features, out_features, kernel_size=3, padding=1, padding_mode="same", stride=1),
        # nn.BatchNorm2d(out_features),
        nn.ReLU()
    )

def conv3x3x3_x2(in_features, middle_features, out_features):
    return nn.Sequential(
        nn.Conv3d(in_features, middle_features, kernel_size=3, padding=1, padding_mode="same", stride=1),
        # nn.BatchNorm3d(middle_features),
        nn.ReLU(),
        nn.Conv3d(middle_features, out_features, kernel_size=3, padding=1, padding_mode="same", stride=1),
        # nn.BatchNorm3d(out_features),
        nn.ReLU()
    )

def conv1x1x1(in_features, out_features):
    return nn.Conv3d(in_features, out_features, kernel_size=1, padding=0, stride=1)

def convT3d(dilation, in_features, out_features):
    return nn.Sequential(
        nn.ConvTranspose3d(in_features, out_features, kernel_size=dilation, stride=dilation, padding=0, output_padding=0),
        # nn.BatchNorm3d(out_features),
        nn.ReLU()
    )

class AuditoryPathway(nn.Module):
    def __init__(self):
        super(AuditoryPathway, self).__init__()

        self.l1_r = defconv3x3_x2(1, 8, 32)
        self.l1_l = defconv3x3_x2(1, 8, 32)
        self.l1_u = defconv3x3_x2(1, 8, 32)
        self.l1_d = defconv3x3_x2(1, 8, 32)

        self.l2_rl = defconv3x3_x2(64, 64, 64)
        self.l2_ud = defconv3x3_x2(64, 64, 64)

        self.l3 = defconv3x3_x2(256, 256, 256)
        
    def forward(self, x):
        B = x.shape[0]
        assert x.shape == (B, 4, 256, 256)

        # Assumption: spectrogram images are stored right, left, up, down order

        r = self.l1_r(x[:,0:1])
        l = self.l1_l(x[:,1:2])
        u = self.l1_u(x[:,2:3])
        d = self.l1_d(x[:,3:4])

        rl = self.l2_rl(circle_c(r, l))
        ud = self.l2_ud(circle_c(u, d))

        r = slash4(r)
        l = slash4(l)
        u = slash4(u)
        d = slash4(d)

        rl = slash4(rl)
        ud = slash4(ud)

        rlud = self.l3(circle_c(r, rl, l, u, ud, d))

        r = slash4(r)
        l = slash4(l)
        u = slash4(u)
        d = slash4(d)

        rl = slash4(rl)
        ud = slash4(ud)

        rlud = slash4(rlud)

        return circle_c(r, rl, r, rlud, u, ud, d)

class BatGNet(nn.Module):
    def __init__(self, debug_mode=False):
        super(BatGNet, self).__init__()
        
        self._debug_mode = debug_mode

        self.lw_ap = AuditoryPathway()
        self.sw_ap = AuditoryPathway()
        
        self.l4 = conv3x3_x2(1024, 512, 256)

        self.fc = nn.Sequential(
            Reshape((256, 4, 4), (256*4*4,)),
            nn.Linear(256*4*4, 64*4*4*4),
            Reshape((64*4*4*4,), (64, 4, 4, 4))
        )

        self.decoder = nn.Sequential(
            convT3d(2, 64, 64),
            conv3x3x3_x2(64, 32, 32),
            convT3d(2, 32, 32),
            conv3x3x3_x2(32, 16, 16),
            convT3d(4, 16, 16),
            conv3x3x3_x2(16, 8, 8),
            conv1x1x1(8, 1),
            Reshape((1, 64, 64, 64), (64, 64, 64))
        )
        
    def forward(self, x):
        if self._debug_mode:
            with torch.autograd.detect_anomaly():
                return self.forward_impl(x)
        else:
            return self.forward_impl(x)
    
    def forward_impl(self, x):
        x = x["input"]
        B = x.shape[0]
        assert x.shape == (B, 8, 256, 256)
        
        # Assumption: spectrogram images are stored LW right, LW left, LW up, LW down, SW right, SW left, SW up, SW down

        lw = self.lw_ap(x[:,0:4])
        sw = self.sw_ap(x[:,4:8])

        x = circle_c(lw, sw)

        x = self.l4(x)

        x = slash4(x)

        x = self.fc(x)

        x = self.decoder(x)

        assert x.shape == (B, 64, 64, 64)

        x = torch.softmax(x, dim=1)

        return DeviceDict({"output": x })


