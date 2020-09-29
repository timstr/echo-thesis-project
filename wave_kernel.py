import torch
import torch.nn as nn
import numpy as np
import math

def make_wave_kernel(propagation_speed, time_step, velocity_damping, velocity_dispersion):
    identity = np.asarray([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])

    laplacian = np.asarray([
        [0.25,  0.50, 0.25],
        [0.50, -3.00, 0.50],
        [0.25,  0.50, 0.25]
    ])

    average = np.asarray([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]) / 9.0

    # identity = np.asarray([
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0]
    # ])

    # laplacian = np.asarray([
    #     [0.0, 0.1,  0.2, 0.1, 0.0],
    #     [0.1, 0.2,  0.4, 0.2, 0.1],
    #     [0.2, 0.4, -4.0, 0.4, 0.2],
    #     [0.1, 0.2,  0.4, 0.2, 0.1],
    #     [0.0, 0.1,  0.2, 0.1, 0.0]
    # ])

    kD = math.pow(velocity_damping, time_step)

    pos_to_pos = identity

    pos_to_vel = (propagation_speed * propagation_speed * laplacian) * time_step * kD

    vel_to_pos = identity * time_step

    vel_to_vel = ((1.0 - velocity_dispersion) * identity + velocity_dispersion * average) * kD

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
        padding=0,
        padding_mode='constant',
        bias=False
    )
    conv.weight = nn.Parameter(kernel, requires_grad=False)
    conv = conv.cuda()
    return conv


def pad_field(field, params, with_grad=False):
    # TODO: use cells one over as well (i.e. use 3 or 6 neighbouring cells)
    # These should probably be symmetric, and left/right or top/bottom weights
    # should be reusable.
    # This will also require padding v0 and v1 (as used in combine() below)
    # with one value on either end (ideally using extension instead of zero-padding)

    assert(len(params.shape) == 1)
    assert(params.shape[0] == 32)

    def apply(v):
        # dim 0: 1 (batch size)
        # dim 1: 2 (position/velocity channels)
        # dim 2: 2 (width of border)
        # dim 3: unconstrained (the interesting data)
        inshape = v.shape
        assert(len(inshape) == 4)
        assert(inshape[0] == 1)
        assert(inshape[1] == 2)
        assert(inshape[2] == 2)

        # output vector
        out = torch.zeros(1, 2, inshape[3]).cuda()
        if not with_grad:
            out = out.detach()

        # repeat-pad to create data for front and back of tensor
        front = v[:, :, :, :1]
        back  = v[:, :, :, -1:]
        v = torch.cat((front, v, back), dim=3)

        param_shape = (1, 2, 2, 1)

        params_center_pos_l = params[0:4].reshape(param_shape)
        params_center_pos_q = params[4:8].reshape(param_shape)
        params_offset_pos_l = params[8:12].reshape(param_shape)
        params_offset_pos_q = params[12:16].reshape(param_shape)

        params_center_vel_l = params[16:20].reshape(param_shape)
        params_center_vel_q = params[20:24].reshape(param_shape)
        params_offset_vel_l = params[24:28].reshape(param_shape)
        params_offset_vel_q = params[28:32].reshape(param_shape)

        v_offneg = v[:, :, :, :-2]
        v_center = v[:, :, :, 1:-1]
        v_offpos = v[:, :, :, 2:]

        out[:, 0, :] = torch.sum(torch.sum((
            params_offset_pos_l * v_offneg + params_offset_pos_q * v_offneg**2 + 
            params_center_pos_l * v_center + params_center_pos_q * v_center**2 + 
            params_offset_pos_l * v_offneg + params_offset_pos_q * v_offneg**2
        ), dim=1), dim=1)
        out[:, 1, :] = torch.sum(torch.sum((
            params_offset_vel_l * v_offneg + params_offset_vel_q * v_offneg**2 + 
            params_center_vel_l * v_center + params_center_vel_q * v_center**2 + 
            params_offset_vel_l * v_offneg + params_offset_vel_q * v_offneg**2
        ), dim=1), dim=1)

        assert(out.shape == (1, 2, inshape[3]))

        return out

    f = field
    top    = apply(f[:, :, :2, :]).unsqueeze(2)
    bottom = apply(f[:, :, -2:, :].flip([2])).unsqueeze(2)

    f = torch.cat((top, f, bottom), dim=2)

    left   = apply(f[:, :, :, :2].permute([0, 1, 3, 2])).unsqueeze(2).permute([0, 1, 3, 2])
    right  = apply(f[:, :, :, -2:].permute([0, 1, 3, 2]).flip([2])).unsqueeze(2).permute([0, 1, 3, 2])

    f = torch.cat((left, f, right), dim=3)

    return f