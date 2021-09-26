import torch
import torch.nn as nn

from assert_eq import assert_eq
from reshape_layer import Reshape


def _c(*args):
    return torch.cat(args, dim=1)


def _slash4(x=None):
    if x is None:
        return nn.MaxPool2d(kernel_size=4, stride=4)
    else:
        return nn.functional.max_pool2d(
            x,
            kernel_size=4,
            stride=4,
        )


def _3x3_2def(in_features, middle_features, out_features):
    return nn.Sequential(
        nn.Conv2d(
            in_features,
            middle_features,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="replicate",
        ),
        nn.BatchNorm2d(middle_features),
        nn.ReLU(),
        nn.Conv2d(
            middle_features,
            out_features,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="replicate",
        ),
        nn.BatchNorm2d(out_features),
        nn.ReLU(),
    )


def _3x3_2conv2d(in_features, middle_features, out_features):
    return nn.Sequential(
        nn.Conv2d(
            in_features,
            middle_features,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
            stride=1,
        ),
        nn.BatchNorm2d(middle_features),
        nn.ReLU(),
        nn.Conv2d(
            middle_features,
            out_features,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
            stride=1,
        ),
        nn.BatchNorm2d(out_features),
        nn.ReLU(),
    )


def _3x3x3_2conv3d(in_features, middle_features, out_features):
    return nn.Sequential(
        nn.Conv3d(
            in_features,
            middle_features,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
            stride=1,
        ),
        nn.BatchNorm3d(middle_features),
        nn.ReLU(),
        nn.Conv3d(
            middle_features,
            out_features,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
            stride=1,
        ),
        nn.BatchNorm3d(out_features),
        nn.ReLU(),
    )


def _1x1x1_conv3d(in_features, out_features):
    return nn.Conv3d(in_features, out_features, kernel_size=1, padding=0, stride=1)


def _x2_convT3d(in_features, out_features):
    return nn.Sequential(
        nn.ConvTranspose3d(
            in_features,
            out_features,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        ),
        nn.BatchNorm3d(out_features),
        nn.ReLU(),
    )


def _x4_convT3d(in_features, out_features):
    return nn.Sequential(
        nn.ConvTranspose3d(
            in_features,
            out_features,
            kernel_size=4,
            stride=4,
            padding=0,
            output_padding=0,
        ),
        nn.BatchNorm3d(out_features),
        nn.ReLU(),
    )


class AuditoryPathway(nn.Module):
    def __init__(self):
        super(AuditoryPathway, self).__init__()

        self.l1_r = _3x3_2def(1, 32, 32)
        self.l1_l = _3x3_2def(1, 32, 32)
        self.l1_u = _3x3_2def(1, 32, 32)
        self.l1_d = _3x3_2def(1, 32, 32)

        self.l2_rl = _3x3_2def(64, 64, 64)
        self.l2_ud = _3x3_2def(64, 64, 64)

        self.l3 = _3x3_2def(256, 256, 256)

    def forward(self, x):
        B = x.shape[0]
        assert_eq(x.shape, (B, 4, 256, 256))

        # Assumption: spectrogram images are stored right, left, up, down order

        r = self.l1_r(x[:, 0:1])
        l = self.l1_l(x[:, 1:2])
        u = self.l1_u(x[:, 2:3])
        d = self.l1_d(x[:, 3:4])

        assert_eq(r.shape, (B, 32, 256, 256))
        assert_eq(l.shape, (B, 32, 256, 256))
        assert_eq(u.shape, (B, 32, 256, 256))
        assert_eq(d.shape, (B, 32, 256, 256))

        rl = self.l2_rl(_c(r, l))
        ud = self.l2_ud(_c(u, d))

        assert_eq(rl.shape, (B, 64, 256, 256))
        assert_eq(ud.shape, (B, 64, 256, 256))

        r = _slash4(r)
        l = _slash4(l)
        u = _slash4(u)
        d = _slash4(d)

        assert_eq(r.shape, (B, 32, 64, 64))
        assert_eq(l.shape, (B, 32, 64, 64))
        assert_eq(u.shape, (B, 32, 64, 64))
        assert_eq(d.shape, (B, 32, 64, 64))

        rl = _slash4(rl)
        ud = _slash4(ud)

        assert_eq(rl.shape, (B, 64, 64, 64))
        assert_eq(ud.shape, (B, 64, 64, 64))

        rlud = self.l3(_c(r, rl, l, u, ud, d))

        assert_eq(rlud.shape, (B, 256, 64, 64))

        r = _slash4(r)
        l = _slash4(l)
        u = _slash4(u)
        d = _slash4(d)

        assert_eq(r.shape, (B, 32, 16, 16))
        assert_eq(l.shape, (B, 32, 16, 16))
        assert_eq(u.shape, (B, 32, 16, 16))
        assert_eq(d.shape, (B, 32, 16, 16))

        rl = _slash4(rl)
        ud = _slash4(ud)

        assert_eq(rl.shape, (B, 64, 16, 16))
        assert_eq(ud.shape, (B, 64, 16, 16))

        rlud = _slash4(rlud)

        assert_eq(rlud.shape, (B, 256, 16, 16))

        output = _c(r, rl, r, rlud, u, ud, d)

        assert_eq(output.shape, (B, 512, 16, 16))

        return output


class BatGNet(nn.Module):
    def __init__(self):
        super(BatGNet, self).__init__()

        self.lw_ap = AuditoryPathway()
        self.sw_ap = AuditoryPathway()

        self.l4 = _3x3_2conv2d(1024, 512, 256)

        self.fc = nn.Sequential(
            Reshape((256, 4, 4), (256 * 4 * 4,)),
            nn.Linear(256 * 4 * 4, 64 * 4 * 4 * 4),
            Reshape((64 * 4 * 4 * 4,), (64, 4, 4, 4)),
        )

        self.decoder = nn.Sequential(
            _x2_convT3d(64, 64),
            _3x3x3_2conv3d(64, 32, 32),
            _x2_convT3d(32, 32),
            _3x3x3_2conv3d(32, 16, 16),
            _x4_convT3d(16, 16),
            _3x3x3_2conv3d(16, 8, 8),
            _1x1x1_conv3d(8, 1),
            Reshape((1, 64, 64, 64), (64, 64, 64)),
        )

    def forward(self, x):
        B = x.shape[0]
        assert_eq(x.shape, (B, 8, 256, 256))

        # Assumption: spectrogram images are stored LW right, LW left, LW up, LW down, SW right, SW left, SW up, SW down

        lw = self.lw_ap(x[:, 0:4])
        sw = self.sw_ap(x[:, 4:8])

        assert_eq(lw.shape, (B, 512, 16, 16))
        assert_eq(sw.shape, (B, 512, 16, 16))

        x = _c(lw, sw)

        assert_eq(x.shape, (B, 1024, 16, 16))

        x = self.l4(x)

        assert_eq(x.shape, (B, 256, 16, 16))

        x = _slash4(x)

        assert_eq(x.shape, (B, 256, 4, 4))

        x = self.fc(x)

        assert_eq(x.shape, (B, 64, 4, 4, 4))

        x = self.decoder(x)

        assert_eq(x.shape, (B, 64, 64, 64))

        # x = torch.sigmoid(x)

        return x
