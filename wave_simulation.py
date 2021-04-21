import torch
import math

from the_device import the_device


def poly(x, n, dim):
    return torch.cat([x ** (i + 1) for i in range(n)], dim=dim)


half_pad_kernel_optimized = torch.tensor(
    [
        [
            [[3.1282861e-02, 3.3955164e-02], [2.3818469e-02, 2.9856248e-02]],
            [[-3.1680989e-01, -2.9902002e-01], [-5.0570101e-02, -3.9272599e-02]],
            [[-3.1371149e-03, -3.3632137e-03], [-2.1783063e-04, -5.0264568e-04]],
            [[2.9526532e-04, 4.0946100e-04], [2.3401692e-03, 2.6417202e-03]],
        ],
        [
            [[-3.0274855e-02, -3.2344308e-02], [-2.3409428e-02, -2.9283607e-02]],
            [[3.2946691e-01, 3.1111991e-01], [4.8102032e-02, 3.7372373e-02]],
            [[2.9055725e-03, 3.1556420e-03], [-1.4176522e-04, 1.6912841e-04]],
            [[-3.0772696e-04, -4.2176869e-04], [-2.4683629e-03, -2.7419692e-03]],
        ],
        [
            [[9.4796732e-02, 5.2073259e-02], [7.6103486e-02, 8.2684182e-02]],
            [[3.4834972e-01, 2.5711864e-01], [2.2886337e-01, 1.8613230e-01]],
            [[3.8627663e-03, 4.9315249e-03], [6.2747821e-03, 2.8311080e-04]],
            [[-2.1003217e-02, 7.5548342e-03], [-6.1300849e-03, -4.5999588e-04]],
        ],
        [
            [[9.2141546e-02, 7.4591033e-02], [9.5857181e-02, 1.2351751e-01]],
            [[2.9001239e-01, 2.2620440e-01], [2.0642446e-01, 1.8981107e-01]],
            [[9.7290548e-03, 4.4364929e-03], [9.6077397e-03, -4.0058928e-04]],
            [[-1.3560701e-02, 7.6356144e-03], [5.1978737e-04, -3.3461576e-05]],
        ],
    ],
    dtype=torch.float32,
).to(the_device)


def pad_fields(field_now, field_prev, half_pad_kernel=None):
    if half_pad_kernel is None:
        half_pad_kernel = half_pad_kernel_optimized
        # half_pad_kernel = torch.zeros(4, 4, 2, 2).to(the_device)
    fields = torch.stack((field_now, field_prev), dim=0)
    order = 2
    c, h, w = fields.shape
    assert c == 2
    ko, ki, kh, kw = half_pad_kernel.shape
    assert (ko, ki, kh, kw) == (4, 2 * order, 2, 2)

    kernel = torch.cat((half_pad_kernel, half_pad_kernel[:, :, :, 1:]), dim=3)
    assert kernel.shape == (4, 2 * order, 2, 3)

    ft = fields[:, :2, :]
    fb = fields[:, -2:, :].flip([1])
    ftb = torch.stack((ft, fb), dim=0)
    ftb = poly(ftb, order, dim=1)
    assert ftb.shape == (2, 2 * order, 2, w)
    ftb = torch.nn.functional.conv2d(ftb, kernel, padding=(0, 1))
    assert ftb.shape == (2, 4, 1, w)

    fields = torch.cat(
        (ftb[0].reshape(2, 2, w), fields, ftb[1].reshape(2, 2, w).flip([1])), dim=1
    )

    assert fields.shape == (2, h + 4, w)

    fl = fields[:, :, :2].permute(0, 2, 1)
    fr = fields[:, :, -2:].permute(0, 2, 1).flip([1])
    flr = torch.stack((fl, fr), dim=0)
    flr = poly(flr, order, dim=1)
    assert flr.shape == (2, 2 * order, 2, h + 4)
    flr = torch.nn.functional.conv2d(flr, kernel, padding=(0, 1))
    assert flr.shape == (2, 4, 1, h + 4)

    fields = torch.cat(
        (
            flr[0].reshape(2, 2, h + 4).permute(0, 2, 1),
            fields,
            flr[1].reshape(2, 2, h + 4).permute(0, 2, 1).flip([2]),
        ),
        dim=2,
    )

    assert fields.shape == (2, h + 4, w + 4)

    return fields[0], fields[1]


def step_simulation(field_now, field_prev, half_pad_kernel=None):
    field_now_unpadded = field_now

    field_now, field_prev = pad_fields(field_now, field_prev, half_pad_kernel)
    # field_now, field_prev = pad_fields(field_now, field_prev, half_pad_kernel)

    dx = 1.0
    c = 1.0
    dt = math.sqrt(3.0 / 8.0)
    gamma = c * dt / dx

    # 2nd order
    # field_next = (
    #     2.0 * (1.0 - 2.0 * gamma**2) * field_now[1:-1,1:-1] - field_prev[1:-1,1:-1]
    #     + gamma**2 * (field_now[2:,1:-1] + field_now[:-2,1:-1] + field_now[1:-1,2:] + field_now[1:-1,:-2])
    # )

    # 4th order
    # field_next = (
    #     (2.0 - 5.0 * gamma**2) * field_now[2:-2,2:-2] - field_prev[2:-2,2:-2]
    #     + (4.0 * gamma**2 / 3.0) * (
    #         field_now[3:-1,2:-2] +
    #         field_now[1:-3,2:-2] +
    #         field_now[2:-2,3:-1] +
    #         field_now[2:-2,1:-3]
    #     )
    #     - (1.0 * gamma**2 / 12.0) * (
    #         field_now[4:,2:-2] +
    #         field_now[:-4,2:-2] +
    #         field_now[2:-2,4:] +
    #         field_now[2:-2,:-4]
    #     )
    # )

    # 4th order modified
    field_next = (
        (2.0 - 5.0 * gamma ** 2) * field_now[2:-2, 2:-2]
        - field_prev[2:-2, 2:-2]
        + (4.0 * gamma ** 2 / 6.0)
        * (
            (0.5 * field_now[3:-1, 1:-3])
            + field_now[3:-1, 2:-2]
            + (0.5 * field_now[3:-1, 3:-1])
            + (0.5 * field_now[1:-3, 1:-3])
            + field_now[1:-3, 2:-2]
            + (0.5 * field_now[1:-3, 3:-1])
            + (0.5 * field_now[1:-3, 3:-1])
            + field_now[2:-2, 3:-1]
            + (0.5 * field_now[3:-1, 3:-1])
            + (0.5 * field_now[1:-3, 1:-3])
            + field_now[2:-2, 1:-3]
            + (0.5 * field_now[3:-1, 1:-3])
        )
        - (1.0 * gamma ** 2 / 24.0)
        * (
            (0.5 * field_now[4:, 1:-3])
            + field_now[4:, 2:-2]
            + (0.5 * field_now[4:, 3:-1])
            + (0.5 * field_now[:-4, 1:-3])
            + field_now[:-4, 2:-2]
            + (0.5 * field_now[:-4, 3:-1])
            + (0.5 * field_now[1:-3, 4:])
            + field_now[2:-2, 4:]
            + (0.5 * field_now[3:-1, 4:])
            + (0.5 * field_now[1:-3, :-4])
            + field_now[2:-2, :-4]
            + (0.5 * field_now[3:-1, :-4])
        )
    )

    return field_next, field_now_unpadded
