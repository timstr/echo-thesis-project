import torch
import matplotlib.pyplot as plt


def question_marks(height, width, scale, stagger=True):
    """
    height  : (integer) height of the output image, in pixels
    width   : (integer) width of the output image, in pixels
    scale   : (number) approximate size of each question mark, in pixels
    stagger : (boolean) if true, alternating columns are staggered vertically
    """
    assert isinstance(height, int)
    assert isinstance(width, int)
    assert isinstance(scale, int) or isinstance(scale, float)
    assert isinstance(stagger, bool)

    img = torch.ones(height, width)

    scale *= 3

    y, x = torch.meshgrid(
        torch.linspace(0.0, height - 1, height) / scale,
        torch.linspace(0.0, width - 1, width) / scale,
    )

    x = x + 0.125

    offset = 0.375
    wraps = 4.0

    x0 = (x - offset) * wraps
    x1 = x0 - torch.floor(x0)
    x = x1 / wraps + offset

    if stagger:
        y = y + 0.5 * torch.floor(x0)
    y0 = (y - 0.35) * 3.0
    y1 = y0 - torch.floor(y0)
    y = y1 / 3.0 + 0.35

    thickness = 0.05

    bottom_dot_y = 0.65
    bottom_dot_x = 0.5

    img[(y - bottom_dot_y) ** 2 + (x - bottom_dot_x) ** 2 <= (thickness / 2) ** 2] = 0.0

    vertical_stripe_top_y = 0.5
    vertical_stripe_bottom_y = 0.575
    vertical_stripe_x = 0.5

    img[
        (x >= (vertical_stripe_x - thickness / 2))
        * (x <= (vertical_stripe_x + thickness / 2))
        * (y >= vertical_stripe_top_y)
        * (y <= vertical_stripe_bottom_y)
    ] = 0.0

    hook_outer_radius = 0.09
    hook_inner_radius = hook_outer_radius - thickness

    hook_center_x = 0.5
    hook_center_y = vertical_stripe_top_y - hook_inner_radius

    img[
        (y - hook_center_y) ** 2 + (x - hook_center_x) ** 2 <= hook_outer_radius ** 2
    ] = 0.0
    img[
        (y - hook_center_y) ** 2 + (x - hook_center_x) ** 2 <= hook_inner_radius ** 2
    ] = 1.0

    img[
        (x >= (hook_center_x - hook_outer_radius))
        * (x <= hook_center_x - thickness / 2)
        * (y >= hook_center_y)
        * (y <= (hook_center_y + hook_outer_radius))
    ] = 1.0

    img[
        (x >= (hook_center_x - hook_outer_radius))
        * (x <= hook_center_x)
        * (y >= hook_center_y)
        * (y <= (hook_center_y + hook_inner_radius + thickness / 2))
    ] = 1.0

    img[
        (y - hook_center_y) ** 2
        + (x - hook_center_x + hook_inner_radius + thickness / 2) ** 2
        <= (thickness / 2) ** 2
    ] = 0.0

    img[
        (y - hook_center_y - hook_inner_radius - thickness / 2) ** 2
        + (x - hook_center_x) ** 2
        <= (thickness / 2) ** 2
    ] = 0.0

    img[
        (y - vertical_stripe_bottom_y) ** 2 + (x - vertical_stripe_x) ** 2
        <= (thickness / 2) ** 2
    ] = 0.0

    return img
