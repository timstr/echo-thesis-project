import torch
import torch.nn as nn
import numpy as np

def make_wave_kernel():
    # TODO: understand this better
    # - figure out effective speed of sound (c)
    # - is this anisotropic?
    # - figure out energy loss (make this intuitive and tunable, i.e. for robustness)
    # - How can different surfaces be modeled? i.e. absorption/reflection
    # https://en.wikipedia.org/wiki/Acoustic_wave_equation
    pos_to_pos = np.asarray([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    pos_to_pos = pos_to_pos / np.sum(pos_to_pos)

    pos_to_vel = np.asarray([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])
    pos_to_vel = pos_to_vel * -0.07#  / np.sum(pos_to_vel)

    vel_to_pos = np.asarray([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    vel_to_pos = vel_to_pos / np.sum(vel_to_pos)

    vel_to_vel = np.asarray([
        [0, 1,  0],
        [1, 10, 1],
        [0, 1,  0]
    ])
    vel_to_vel = vel_to_vel / np.sum(vel_to_vel)

    kernel = torch.Tensor(
        [
            [pos_to_pos, vel_to_pos],
            [pos_to_vel, vel_to_vel]
        ]
    )
    # Padding mode can be 'constant', 'reflect', 'replicate', or 'circular'
    conv = nn.Conv2d(
        in_channels=2,
        out_channels=2,
        kernel_size=3,
        stride=1,
        padding=1,
        padding_mode='constant',
        bias=False
    )
    conv.weight = nn.Parameter(kernel, requires_grad=False)
    conv = conv.cuda()
    return conv