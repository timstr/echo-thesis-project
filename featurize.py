from EchoLearnNN import EchoLearnNN
from config import OutputConfig, wavesim_duration
import torch
import math
import numpy as np
import random
import itertools

from shape_types import CIRCLE, RECTANGLE
from the_device import the_device, what_my_gpu_can_handle
from device_dict import DeviceDict

def all_yx_locations(size):
    ls = torch.linspace(0, 1, size).to(the_device)
    return torch.stack(
        torch.meshgrid(ls, ls),
        dim=2
    ).reshape(size**2, 2).permute(1, 0)

def all_zyx_locations(size):
    ls = torch.linspace(0, 1, size).to(the_device)
    zyx = torch.stack(
        torch.meshgrid((ls, ls, ls)),
        dim=3
    )
    zyx = zyx.reshape(size**3, 3).permute(1, 0)
    return zyx


def cutout_circle(barrier, y, x, rad):
    h, w = barrier.shape
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, h),
            torch.linspace(0, 1, w)
        ),
        dim=-1
    )
    loc = torch.Tensor([[[y, x]]])
    diffs = grid - loc
    distsqr = torch.sum(diffs**2, dim=-1)
    mask = (distsqr < rad**2)
    barrier[mask] = 0.0

def cutout_rectangle(barrier, y, x, height, width):
    assert len(barrier.shape) == 2
    h, w = barrier.shape
    y0 = int(h * (y - height/2))
    y1 = int(h * (y + height/2))
    x0 = int(w * (x - width/2))
    x1 = int(w * (x + width/2))
    barrier[y0:y1, x0:x1] = 0.0

def cutout_square(barrier, y, x, size, angle):
    h, w = barrier.shape
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, h),
            torch.linspace(0, 1, w)
        ),
        dim=-1
    )
    loc = torch.Tensor([[[y, x]]])
    disps = grid - loc
    c = math.cos(angle)
    s = math.sin(angle)
    rotmat = torch.tensor([
        [ c, s]
        [-s, c]
    ], dtype=torch.float)
    disps_rot = torch.matmul(disps, rotmat)
    halfsize = size / 2
    mask = (disps_rot[:,:,0] >= halfsize) * (disps_rot[:,:,1] >= halfsize)
    barrier[mask] = 0.0

def sdf_batch_circle(coordinates_yx_batch, circle_y, circle_x, circle_r):
    assert len(coordinates_yx_batch.shape) == 2
    assert coordinates_yx_batch.shape[0] == 2
    dev = coordinates_yx_batch.device
    circle_yx = torch.tensor([circle_y, circle_x], dtype=torch.float).unsqueeze(-1).to(dev)
    return torch.sqrt(torch.sum((coordinates_yx_batch - circle_yx)**2, dim=0)) - circle_r

def sdf_batch_rectangle(coordinates_yx_batch, rect_y, rect_x, rect_h, rect_w, rect_angle):
    assert len(coordinates_yx_batch.shape) == 2
    assert coordinates_yx_batch.shape[0] == 2
    dev = coordinates_yx_batch.device
    rect_yx = torch.tensor([rect_y, rect_x], dtype=torch.float).unsqueeze(-1).to(dev)

    disps = coordinates_yx_batch - rect_yx
    c = math.cos(-rect_angle)
    s = math.sin(-rect_angle)
    rot_y = c * disps[0,:] - s * disps[1,:]
    rot_x = s * disps[0,:] + c * disps[1,:]
    disps_rot = torch.stack((rot_y, rot_x), dim=0)
    abs_disps_rot = torch.abs(disps_rot)

    rect_hw_halved = torch.tensor([rect_h * 0.5, rect_w * 0.5], dtype=torch.float).unsqueeze(-1).to(dev)
    edgeDisp = torch.abs(abs_disps_rot) - rect_hw_halved
    outerDist = torch.sqrt(torch.sum(torch.clamp(edgeDisp, min=0.0)**2, dim=0))
    innerDist = torch.clamp(torch.max(edgeDisp, dim=0)[0], max=0.0)
    return outerDist + innerDist

def sdf_batch_one_shape(coordinates_yx_batch, shape_tuple):
    ty, y, x = shape_tuple[:3]
    params = shape_tuple[3:]
    if ty == CIRCLE:
        r, = params
        return sdf_batch_circle(coordinates_yx_batch, y, x, r)
    elif ty == RECTANGLE:
        h, w, a = params
        return sdf_batch_rectangle(coordinates_yx_batch, y, x, h, w, a)
    else:
        raise Exception("Unknown shape type")

def sdf_batch(coordinates_yx_batch, obs):
    assert len(obs) > 0
    vals = sdf_batch_one_shape(coordinates_yx_batch, obs[0])
    for o in obs[1:]:
        vals = torch.min(
            torch.stack((
                vals,
                sdf_batch_one_shape(coordinates_yx_batch, o)
            ), dim=1),
            dim=1
        )[0]
    return vals

def heatmap_batch_circle(coordinates_yx_batch, circle_y, circle_x, circle_r):
    assert len(coordinates_yx_batch.shape) == 2
    assert coordinates_yx_batch.shape[0] == 2
    dev = coordinates_yx_batch.device
    circle_yx = torch.tensor([circle_y, circle_x], dtype=torch.float).unsqueeze(-1).to(dev)
    dists = torch.sum((coordinates_yx_batch - circle_yx)**2, dim=0)
    ret = torch.zeros(coordinates_yx_batch.shape[1])
    ret[dists <= circle_r**2] = 1.0
    return ret

def heatmap_batch_rectangle(coordinates_yx_batch, rect_y, rect_x, rect_h, rect_w, rect_angle):
    assert len(coordinates_yx_batch.shape) == 2
    assert coordinates_yx_batch.shape[0] == 2
    dev = coordinates_yx_batch.device
    rect_yx = torch.tensor([rect_y, rect_x], dtype=torch.float).unsqueeze(-1).to(dev)
    disps = coordinates_yx_batch - rect_yx

    c = math.cos(-rect_angle)
    s = math.sin(-rect_angle)
    rot_y = c * disps[0,:] - s * disps[1,:]
    rot_x = s * disps[0,:] + c * disps[1,:]
    disps_rot = torch.stack((rot_y, rot_x), dim=0)
    abs_disps_rot = torch.abs(disps_rot)

    ret = torch.zeros(coordinates_yx_batch.shape[1])
    inside_y = abs_disps_rot[0] < (rect_h * 0.5)
    inside_x = abs_disps_rot[1] < (rect_w * 0.5)
    ret[inside_y * inside_x] = 1.0
    return ret

def heatmap_batch_one_shape(coordinates_yx_batch, shape_tuple):
    ty, y, x = shape_tuple[:3]
    params = shape_tuple[3:]
    if ty == CIRCLE:
        r, = params
        return heatmap_batch_circle(coordinates_yx_batch, y, x, r)
    elif ty == RECTANGLE:
        h, w, a = params
        return heatmap_batch_rectangle(coordinates_yx_batch, y, x, h, w, a)
    else:
        raise Exception("Unknown shape type")

def heatmap_batch(coordinates_yx_batch, obs):
    assert len(obs) > 0
    vals = heatmap_batch_one_shape(coordinates_yx_batch, obs[0])
    for o in obs[1:]:
        vals = torch.max(
            torch.stack((
                vals,
                heatmap_batch_one_shape(coordinates_yx_batch, o)
            ), dim=1),
            dim=1
        )[0]
    return vals

def intersect_line_segment(ry_rx_dy_dx, p1y, p1x, p2y, p2x, no_collision_dist):
    """
        ry, rx   : ray (y,x) origin
        dy, dx   : ray (y,x) direction (expected to be a unit vector)
        p1y, p1x : point (y,x) for first end of line segment
        p2y, p2x : point (y,x) for second end of line segment
    """
    assert len(ry_rx_dy_dx.shape) == 2
    assert ry_rx_dy_dx.shape[0] == 4
    N = ry_rx_dy_dx.shape[1]

    ry = ry_rx_dy_dx[0]
    rx = ry_rx_dy_dx[1]
    dy = ry_rx_dy_dx[2]
    dx = ry_rx_dy_dx[3]

    eps = 1e-6

    v1y = ry - p1y
    v1x = rx - p1x
    v2x = p2x - p1x
    v2y = p2y - p1y
    v3x = -dy
    v3y = dx

    v2_dot_v3 = (v2x * v3x) + (v2y * v3y)

    parallel = torch.abs(v2_dot_v3) < eps
    

    v1_dot_v3 = (v1x * v3x) + (v1y * v3y)
    t2 = v1_dot_v3 / v2_dot_v3

    outside_segment = ~((t2 >= 0.0) * (t2 <= 1.0))
    

    v2_cross_v1 = v2x * v1y - v2y * v1x
    t1 = v2_cross_v1 / v2_dot_v3

    out = t1
    assert out.shape == (N,)
    out[parallel] = no_collision_dist
    out[outside_segment] = no_collision_dist
    return out

def intersect_circle(ry_rx_dy_dx, sy, sx, sr, no_collision_dist):
    """
        ry, rx : ray (y,x) origin
        dy, dx : ray (y,x) direction (expected to be a unit vector)
        sy, sx : shape y,x e.g. center of circle
        sr     : shape radius
    """
    assert len(ry_rx_dy_dx.shape) == 2
    assert ry_rx_dy_dx.shape[0] == 4
    N = ry_rx_dy_dx.shape[1]
    
    ry = ry_rx_dy_dx[0]
    rx = ry_rx_dy_dx[1]
    dy = ry_rx_dy_dx[2]
    dx = ry_rx_dy_dx[3]

    a = dy**2 + dx**2
    b = 2.0 * (dy * (ry - sy) + dx * (rx - sx))
    c = (ry - sy)**2 + (rx - sx)**2 - sr**2

    d = b**2 - 4.0 * a * c

    no_solution = d < 0.0

    sqrtd = torch.sqrt(d)

    t = (-b - sqrtd) / (2.0 * a)
    t1 = (-b + sqrtd) / (2.0 * a)

    # Assumption: ray is not inside circle
    t1_lt_t = t1 < t
    t[t1_lt_t] = t1[t1_lt_t]

    t[no_solution] = no_collision_dist

    return t

def intersect_rectangle(ry_rx_dy_dx, sy, sx, sh, sw, sa, no_collision_dist):
    """
        ry, rx : ray (y,x) origin
        dy, dx : ray (y,x) direction (expected to be a unit vector)
        sy, sx : shape (y,x) e.g. center of rectangle
        sh, sw : shape height and width
        sa     : shape angle
    """
    assert len(ry_rx_dy_dx.shape) == 2
    assert ry_rx_dy_dx.shape[0] == 4
    N = ry_rx_dy_dx.shape[1]
    
    s = math.sin(sa)
    c = math.cos(sa)
    hsh = 0.5 * sh
    hsw = 0.5 * sw

    khy = c * hsh
    khx = s * hsh

    kwy = s * hsw
    kwx = c * hsw

    tl = (sy - khy + kwy, sx - kwx - khx)
    tr = (sy - khy - kwy, sx + kwx - khx)
    br = (sy + khy - kwy, sx + kwx + khx)
    bl = (sy + khy + kwy, sx - kwx + khx)
    
    t_vals = torch.stack((
        intersect_line_segment(ry_rx_dy_dx, *tl, *tr, no_collision_dist),
        intersect_line_segment(ry_rx_dy_dx, *tr, *br, no_collision_dist),
        intersect_line_segment(ry_rx_dy_dx, *br, *bl, no_collision_dist),
        intersect_line_segment(ry_rx_dy_dx, *bl, *tl, no_collision_dist),
    ), dim=0)

    min_t = torch.min(t_vals, dim=0)[0]

    assert min_t.shape == (N,)

    return min_t

def distance_along_line_of_sight(obs, ry_rx_dy_dx, no_collision_dist):
    """
        obs    : list of obstacles
        ry, rx : (y,x) ray origin
        dy, dx : (y,x) ray direction
    """
    assert len(ry_rx_dy_dx.shape) == 2
    assert ry_rx_dy_dx.shape[0] == 4
    N = ry_rx_dy_dx.shape[1]

    out = torch.ones((N,), device=the_device) * no_collision_dist

    for shape_params in obs:
        ty = shape_params[0]
        rest = shape_params[1:]
        if ty == CIRCLE:
            assert len(rest) == 3
            d = intersect_circle(ry_rx_dy_dx, *rest, no_collision_dist)
        elif ty == RECTANGLE:
            assert len(rest) == 5
            d = intersect_rectangle(ry_rx_dy_dx, *rest, no_collision_dist)
        else:
            raise Exception("Unrecognized shape")
        d_lt_out = d < out
        out[d_lt_out] = d[d_lt_out]
    return out

def lines_of_sight_from_bottom_up(obs, values):
    assert len(values.shape) == 1
    N = values.shape[0]
    ry_rx_dy_dx = torch.stack((
        torch.ones(N),
        values,
        -torch.ones(N),
        torch.zeros(N)
    ), dim=0)
    assert ry_rx_dy_dx.shape == (4, N)
    ry_rx_dy_dx = ry_rx_dy_dx.to(the_device)
    return distance_along_line_of_sight(obs, ry_rx_dy_dx, 1.0)

def make_image_pred(example, img_size, network, num_splits, predict_variance):
    num_splits = max(num_splits, 1)
    num_dims = 2 if predict_variance else 1
    num_samples = img_size**2
    outputs = torch.zeros(num_dims, num_samples)
    assert (num_samples % num_splits) == 0
    split_size = num_samples // num_splits
    xy_locations = all_yx_locations(img_size).permute(1, 0).unsqueeze(0)
    d = DeviceDict({
        "input": example["input"][:1],
    })
    for i in range(num_splits):
        begin = i * split_size
        end = (i + 1) * split_size
        d["params"] = xy_locations[:, begin:end]
        pred = network(d)["output"]
        assert pred.shape == (1, num_dims, split_size)
        outputs[:, begin:end] = pred.reshape(num_dims, split_size).detach()
        pred = None
    return outputs.reshape(num_dims, img_size, img_size).detach().cpu()

def make_volume_pred(example, img_size, network, num_splits, predict_variance):
    num_splits = max(num_splits, 1)
    num_dims = 2 if predict_variance else 1
    num_samples = img_size**3
    outputs = torch.zeros(num_dims, num_samples)
    assert (num_samples % num_splits) == 0
    split_size = num_samples // num_splits
    xyz_locations = all_zyx_locations(img_size).permute(1, 0).unsqueeze(0)
    d = DeviceDict({
        "input": example["input"][:1],
    })
    for i in range(num_splits):
        begin = i * split_size
        end = (i + 1) * split_size
        d["params"] = xyz_locations[:, begin:end]
        pred = network(d)["output"]
        assert pred.shape == (1, num_dims, split_size)
        outputs[:, begin:end] = pred.reshape(num_dims, split_size).detach()
        pred = None
    return outputs.reshape(num_dims, img_size, img_size, img_size).detach().cpu()

def make_sdf_image_pred(example, img_size, network, num_splits, predict_variance):
    return make_image_pred(example, img_size, network, num_splits, predict_variance)

def make_sdf_image_gt(example, img_size):
    obs = example["obstacles_list"][0]

    coordinates_yx = all_yx_locations(img_size)

    return sdf_batch(coordinates_yx, obs).reshape(img_size, img_size)

def smoothstep(edge0, edge1, x):
    assert edge0 < edge1
    t = torch.clamp((x - edge0) / (edge1 - edge0), min=0.0, max=1.0)
    return t * t * (3.0 - 2.0 * t)

def inigo_quilez_sdf_colours(img):
    # adapted from https://www.shadertoy.com/view/XdXcRB
    assert len(img.shape) == 2
    img = img.unsqueeze(2) * 2.0 # Lazy hack to make things appear more detailed
    base_white = torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float, device=img.device)
    base_mod = torch.tensor([[[0.1, 0.4, 0.7]]], dtype=torch.float, device=img.device)
    col = base_white - torch.sign(img) * base_mod
    col *= 1.0 - torch.exp(-2.0 * torch.abs(img))
    col *= 0.8 + 0.2 * torch.cos(60.0 * img) # Note: changed from original 140.0 to make lines less dense
    col = torch.lerp(col, base_white, 1.0 - smoothstep(0.0, 0.02, torch.abs(img)))
    return col

def blue_orange_sdf_colours(img):
    H, W = img.shape
    img = img.unsqueeze(2)

    def colour(r, g, b):
        return torch.tensor([r, g, b], dtype=torch.float, device=img.device).reshape(1, 1, 3)
    blue = colour(0.22, 0.33, 0.66)
    orange = colour(0.93, 0.48, 0.10)
    paler_blue = colour(0.50, 0.58, 0.82)
    paler_orange = colour(0.93, 0.87, 0.28)
    white = colour(1.0, 1.0, 1.0)

    sign = torch.sign(img)

    base_colour = blue * (0.5 - 0.5 * sign) + orange * (0.5 + 0.5 * sign)
    paler_colour = paler_blue * (0.5 - 0.5 * sign) + paler_orange * (0.5 + 0.5 * sign)
    mix = torch.exp(-4.0 * torch.abs(img))

    out = base_colour + mix * (paler_colour - base_colour)

    out *= 1.0 - 0.2 * torch.cos(60.0 * img)**4

    out = torch.lerp(out, white, 1.0 - smoothstep(0.0, 0.02, torch.abs(img)))

    return out

def colourize_sdf(img):
    return blue_orange_sdf_colours(img)

def blue_yellow(img):
    assert len(img.shape) == 2
    img = torch.clamp(img, min=0.0, max=1.0).unsqueeze(-1)
    def colour(r, g, b):
        return torch.tensor([r, g, b], dtype=torch.float, device=img.device).reshape(1, 1, 3)
    blue = colour(0.0, 0.59, 1.0)
    yellow = colour(1.0, 1.0, 0.0)
    
    return (1.0 - img) * blue + img * yellow

def red_white_blue_banded(img):
    assert len(img.shape) == 2
    img = img * 5.0
    r = torch.clamp(img + 1.0, min=0.0, max=1.0)
    g = torch.clamp(1.0 - torch.abs(img), min=0.0, max=1.0)
    b = torch.clamp(-img + 1.0, min=0.0, max=1.0)
    x = 0.5 + img * 5.0
    x = torch.clamp(1.0 * torch.abs(1.0 - 2.0 * (x - torch.floor(x))), min=0.0, max=1.0)
    rgb = torch.stack(
        (r, g, b),
        dim=2
    ) * (0.5 + 0.5 * x.unsqueeze(-1))
    z = torch.clamp(1.0 - 20.0 * torch.abs(img), min=0.0, max=1.0).unsqueeze(-1)
    z_color = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float).unsqueeze(0).unsqueeze(0).to(img.device)
    rgb = rgb + (z_color - rgb) * z
    return rgb

def red_white_blue(img):
    assert len(img.shape) == 2
    img = img.unsqueeze(-1)
    close_to_one  = torch.clamp((2.0 * img - 1.0), 0.0, 1.0)
    close_to_half = torch.clamp((1.0 - 2.0 * torch.abs(img - 0.5)), 0.0, 1.0)
    close_to_zero = torch.clamp((1.0 - 2.0 * img), 0.0, 1.0)
    def colour(r, g, b):
        return torch.tensor([r, g, b], dtype=torch.float, device=img.device).reshape(1, 1, 3)
    red   = colour(0.71, 0.01, 0.15)
    white = colour(1.00, 1.00, 1.00)
    blue  = colour(0.23, 0.30, 0.75)
    return close_to_one * red + close_to_half * white + close_to_zero * blue
    
def purple_yellow(img):
    assert len(img.shape) == 2
    img = torch.clamp(img, min=0.0, max=1.0).unsqueeze(-1)
    def colour(r, g, b):
        return torch.tensor([r, g, b], dtype=torch.float, device=img.device).reshape(1, 1, 3)
    purple = colour(0.27, 0.00, 0.33)
    yellow = colour(0.99, 0.91, 0.14)
    
    return (1.0 - img) * purple + img * yellow


def make_heatmap_image_pred(example, img_size, network, num_splits, predict_variance):
    return make_image_pred(example, img_size, network, num_splits, predict_variance)

def make_heatmap_image_gt(example, img_size):
    obs = example["obstacles_list"][0]

    coordinates_yx = all_yx_locations(img_size)

    return torch.clamp(img_size * -sdf_batch(coordinates_yx, obs), 0.0, 1.0).reshape(img_size, img_size)

def make_depthmap_gt(example, img_size):
    obs = example["obstacles_list"][0]
    values = torch.linspace(0.0, 1.0, img_size)
    return lines_of_sight_from_bottom_up(obs, values)

def make_depthmap_pred(example, img_size, network):
    locations = torch.linspace(0, 1, img_size).reshape(1, img_size, 1).to(the_device)
    d = DeviceDict({
        "input": example["input"][:1],
        "params": locations
    })
    pred = network(d)["output"]
    assert len(pred.shape) == 3
    assert pred.shape[0] == 1
    outputDims = pred.shape[1]
    assert pred.shape[2] == img_size
    return pred.reshape(outputDims, img_size).detach().cpu()

def make_echo4ch_heatmap_volume_pred(example, resolution, network, num_splits, predict_variance):
    return make_volume_pred(example, resolution, network, num_splits, predict_variance)

def make_echo4ch_depthmap_image_pred(example, resolution, network, num_splits, predict_variance):
    return make_image_pred(example, resolution, network, num_splits, predict_variance)

def make_echo4ch_dense_implicit_output_pred(example, network, output_config):
    assert isinstance(example, DeviceDict)
    assert isinstance(network, EchoLearnNN)
    assert isinstance(output_config, OutputConfig)
    assert output_config.implicit
    res = output_config.resolution
    var = output_config.predict_variance
    assert res == 64
    if output_config.format == "depthmap":
        num_splits = res**2 // what_my_gpu_can_handle
        return make_echo4ch_depthmap_image_pred(example, res, network, num_splits, var)
    elif output_config.format == "heatmap":
        num_splits = res**3 // what_my_gpu_can_handle
        return make_echo4ch_heatmap_volume_pred(example, res, network, num_splits, var)
    else:
        raise Exception("Unrecognized output representation")

def make_dense_tof_cropped_output_pred(example, network, output_config):
    assert isinstance(example, DeviceDict)
    assert isinstance(network, EchoLearnNN)
    assert isinstance(output_config, OutputConfig)
    assert output_config.tof_cropping
    assert output_config.dims == 2

    res = output_config.resolution
    predict_variance = output_config.predict_variance

    num_splits = max(res**2 // what_my_gpu_can_handle, 1)

    num_output_dims = 2 if predict_variance else 1
    num_samples = res**2
    outputs = torch.zeros(num_samples, num_output_dims)
    assert (num_samples % num_splits) == 0
    split_size = num_samples // num_splits

    xy_locations = all_yx_locations(res).permute(1, 0)
    assert xy_locations.shape == (num_samples, 2)

    input_batch_of_one = example["input"][:1]
    assert len(input_batch_of_one.shape) == 3
    assert input_batch_of_one.shape[2] == wavesim_duration

    d = DeviceDict({
        "input": input_batch_of_one.repeat(split_size, 1, 1),
    })

    for i in range(num_splits):
        begin = i * split_size
        end = (i + 1) * split_size
        d["params"] = xy_locations[begin:end]
        pred = network(d)["output"]
        assert pred.shape == (split_size, num_output_dims)
        outputs[begin:end, :] = pred.detach()
        pred = None

    return outputs.permute(1, 0).reshape(num_output_dims, res, res).detach().cpu()   

def make_dense_implicit_output_pred(example, network, output_config):
    assert isinstance(example, DeviceDict)
    assert isinstance(network, EchoLearnNN)
    assert isinstance(output_config, OutputConfig)
    assert output_config.implicit
    res = output_config.resolution
    var = output_config.predict_variance
    if output_config.format == "sdf":
        num_splits = res**2 // what_my_gpu_can_handle
        return make_sdf_image_pred(example, res, network, num_splits, var)
    elif output_config.format == "heatmap":
        num_splits = res**2 // what_my_gpu_can_handle
        return make_heatmap_image_pred(example, res, network, num_splits, var)
    elif output_config.format == "depthmap":
        return make_depthmap_pred(example, res, network)
    else:
        raise Exception("Unrecognized output representation")

def obstacle_radius(o):
    t = o[0]
    shape = o[3:]
    if t == CIRCLE:
        r, = shape
        return r
    elif t == RECTANGLE:
        h, w = shape[:2]
        return math.hypot(h, w)
    else:
        raise Exception("Unrecognized obstacle type")

def obstacles_too_close(obstacles, min_dist):
    n = len(obstacles)
    bcs = []
    for o in obstacles:
        bcs.append((o[1], o[2], obstacle_radius(o)))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            yi, xi, ri = bcs[i]
            yj, xj, rj = bcs[j]
            d = math.hypot(yi - yj, xi - xj)
            if d - ri - rj < min_dist:
                return True
    return False


def obstacles_occluded(obstacles):
    n = len(obstacles)
    bbs = []
    for o in obstacles:
        x = o[2]
        rad = obstacle_radius(o)
        bbs.append((x - rad, x + rad))
    bbs = sorted(bbs, key=lambda x: x[0])
    for bb1, bb2 in zip(bbs, bbs[1:]):
        if bb1[1] > bb2[0]:
            return True
    return False


def every_possible_obstacle(min_size, max_size, top, bottom, left, right, num_size_increments, num_space_increments, num_angle_increments):
    for size in np.linspace(min_size, max_size, num_size_increments):
        halfsize = size / 2
        for y in np.linspace(top + halfsize, bottom - halfsize, num_space_increments):
            for x in np.linspace(left + halfsize, right - halfsize, num_space_increments):
                yield CIRCLE, y, x, halfsize

                edge_len = (1.0 / math.sqrt(2.0)) * size
                for angle in np.linspace(0.0, 0.5 * np.pi, num_angle_increments, endpoint=False):
                    yield RECTANGLE, y, x, edge_len, edge_len, angle

def all_possible_obstacles(max_num_obstacles, min_dist, min_size, max_size, top, bottom, left, right, num_size_increments, num_space_increments, num_angle_increments):
    for n in range(1, max_num_obstacles + 1):
        prod = itertools.product(*(every_possible_obstacle(
            min_size, max_size,
            top, bottom,
            left, right,
            num_size_increments,
            num_space_increments,
            num_angle_increments
        ) for _ in range(n)))
        for obs in prod:
            if not obstacles_too_close(obs, min_dist):
                yield obs

def make_random_obstacles(max_num_obstacles=10, min_dist=0.05):
    n = random.randint(1, max_num_obstacles)

    obs = []
    
    def collision(shape):
        nonlocal obs
        y1, x1 = shape[1:3]
        r1 = obstacle_radius(shape)
        for s in obs:
            y2, x2 = s[1:3]
            r2 = obstacle_radius(s)
            d = math.hypot(y1 - y2, x1 - x2)
            if d - r1 - r2 < min_dist:
                return True
        return False
    
    for _ in range(n):
        for _ in range(100):
            y = random.uniform(0, 0.75)
            # y = random.uniform(0, 1)
            x = random.uniform(0, 1)
            if random.random() < 0.5:
                # rectangle
                h = random.uniform(0.01, 0.2)
                w = random.uniform(0.01, 0.2)
                r = random.uniform(0.0, np.pi / 2.0)
                o = (RECTANGLE, y, x, h, w, r)
                if collision(o):
                    continue
                obs.append(o)
                break
            else:
                # circle
                r = random.uniform(0.01, 0.1)
                o = (CIRCLE, y, x, r)
                if collision(o):
                    continue
                obs.append(o)
                break
    return obs

def make_implicit_params_train(num, representation):
    dim = 1 if (representation == "depthmap") else 2
    return torch.rand(num, dim)
    
def make_implicit_params_validation(img_size, dims):
    assert dims in [1, 2, 3]
    ls = torch.linspace(0, 1, img_size)
    if dims == 1:
        return ls.reshape(1, img_size, 1)
    elif dims == 2:
        return torch.stack(
            torch.meshgrid((ls, ls)),
            dim=2
        ).to(the_device).reshape(1, img_size**2, 2)
    elif dims == 3:
        return torch.stack(
            torch.meshgrid((ls, ls, ls)),
            dim=2
        ).to(the_device).reshape(1, img_size**3, 3)

def make_implicit_outputs(obs, params, representation):
    assert len(params.shape) == 2
    if representation == "sdf":            
        assert params.shape[1] == 2
        return sdf_batch(params.permute(1, 0), obs)
    elif representation == "heatmap":
        assert params.shape[1] == 2
        return heatmap_batch(params.permute(1, 0), obs)
    elif representation == "depthmap":   
        assert params.shape[1] == 1
        return lines_of_sight_from_bottom_up(obs, params[:,0])
    else:
        raise Exception("Unrecognized representation")

def make_dense_outputs(obs, representation, img_size):
    theDict = DeviceDict({ "obstacles_list": [obs]})
    if representation == "sdf":
        output = make_sdf_image_gt(theDict, img_size)
        assert output.shape == (img_size, img_size)
    elif representation == "heatmap":
        output = make_heatmap_image_gt(theDict, img_size)
        assert output.shape == (img_size, img_size)
    elif representation == "depthmap":
        output = make_depthmap_gt(theDict, img_size)
        assert len(output.shape) == 1
        assert output.shape[0] == img_size
    else:
        raise Exception("Unown representation")
    return output

def sample_dense_output(output, params):
    dims = len(output.shape)
    assert dims in [2, 3]
    assert len(params.shape) == 2
    assert params.shape[1] == dims
    N = params.shape[0]
    if dims == 2:
        H, W = output.shape
        D = None
    elif dims == 3:
        H, W, D = output.shape
    else:
        raise Exception("Unrecognized tensor shape")
    indices_h = torch.floor(params[:,0] * (H - 1)).to(torch.long)
    indices_w = torch.floor(params[:,1] * (W - 1)).to(torch.long)
    indices_flat = W * indices_h + indices_w
    if D is not None:
        indices_d = torch.floor(params[:,2] * (W - 1)).to(torch.long)
        indices_flat = D * indices_flat + indices_d
    indices_flat = indices_flat.to(output.device)
    values = torch.index_select(output.reshape(-1), dim=0, index=indices_flat)
    assert values.shape == (N,)
    return values

def implicit_samples_from_dense_output(arr, num_samples):
    dims = len(arr.shape)
    params = torch.rand((num_samples, dims))
    values = sample_dense_output(arr, params)
    return params, values

def make_deterministic_validation_batches_implicit(example, output_config):
    assert isinstance(output_config, OutputConfig)
    res = output_config.resolution
    fmt = output_config.format
    B = example["input"].shape[0]

    params = make_implicit_params_validation(res, output_config.dims)
    assert params.shape[0] == 1
    data_size = params.shape[1]

    num_splits = max(1, data_size * B // what_my_gpu_can_handle)
    assert data_size % num_splits == 0
    split_size = data_size // num_splits

    batches = []
    for i in range(num_splits):
        begin = i * split_size
        end = (i + 1) * split_size
        output = torch.zeros(B, split_size)
        if output_config.using_echo4ch:
            output_full = example["gt_depthmap"] if (fmt == "depthmap") else example["gt_heatmap"]
            for j in range(B):
                output[j] = sample_dense_output(output_full[0], params[0, begin:end])
        else:
            obsobs = example["obstacles_list"]
            assert len(obsobs) == B
            for j, obs in enumerate(obsobs):
                output[j] = make_implicit_outputs(obs, params[0, begin:end], fmt)
        d = DeviceDict(example.copy())
        d["params"] = params[:, begin:end].repeat(B, 1, 1)
        d["output"] = output
        batches.append(d)
    return batches

