import torch
import math
import scipy.interpolate
import numpy as np
import random
import itertools

from torch import linspace

from the_device import the_device
from device_dict import DeviceDict

CIRCLE = "Circle"
RECTANGLE = "Rectangle"

# TODO:
# - spectrogram
# - ?

# For old simulation (datasets version 1 to 5)
# magic_speed_of_sound = 1.0 / 1050.0

# For dataset version 6+
magic_speed_of_sound = 1.0 / 200.0

dataset_field_size = 512

position_negative = (dataset_field_size // 2 - 10) / dataset_field_size
position_positive = (dataset_field_size // 2 + 10) / dataset_field_size

receiver_locations = [
    (position_negative, position_negative),
    (position_negative, position_positive),
    (position_positive, position_negative),
    (position_positive, position_positive)
]


def all_yx_locations(size):
    ls = torch.linspace(0, 1, size).to(the_device)
    return torch.stack(
        torch.meshgrid(ls, ls),
        dim=2
    ).reshape(size**2, 2).permute(1, 0)


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
    assert(len(barrier.shape) == 2)
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
    ])
    disps_rot = torch.matmul(disps, rotmat)
    halfsize = size / 2
    mask = (disps_rot[:,:,0] >= halfsize) * (disps_rot[:,:,1] >= halfsize)
    barrier[mask] = 0.0

def sdf_batch_circle(coordinates_yx_batch, circle_y, circle_x, circle_r):
    assert(len(coordinates_yx_batch.shape) == 2)
    assert(coordinates_yx_batch.shape[0] == 2)
    dev = coordinates_yx_batch.device
    circle_yx = torch.tensor([circle_y, circle_x]).unsqueeze(-1).to(dev)
    return torch.sqrt(torch.sum((coordinates_yx_batch - circle_yx)**2, dim=0)) - circle_r

def sdf_batch_rectangle(coordinates_yx_batch, rect_y, rect_x, rect_h, rect_w, rect_angle):
    assert(len(coordinates_yx_batch.shape) == 2)
    assert(coordinates_yx_batch.shape[0] == 2)
    dev = coordinates_yx_batch.device
    rect_yx = torch.tensor([rect_y, rect_x]).unsqueeze(-1).to(dev)

    disps = coordinates_yx_batch - rect_yx
    c = math.cos(-rect_angle)
    s = math.sin(-rect_angle)
    rot_y = c * disps[0,:] - s * disps[1,:]
    rot_x = s * disps[0,:] + c * disps[1,:]
    disps_rot = torch.stack((rot_y, rot_x), dim=0)
    abs_disps_rot = torch.abs(disps_rot)

    rect_hw_halved = torch.tensor([rect_h * 0.5, rect_w * 0.5]).unsqueeze(-1).to(dev)
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
    assert(len(obs) > 0)
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
    assert(len(coordinates_yx_batch.shape) == 2)
    assert(coordinates_yx_batch.shape[0] == 2)
    dev = coordinates_yx_batch.device
    circle_yx = torch.tensor([circle_y, circle_x]).unsqueeze(-1).to(dev)
    dists = torch.sum((coordinates_yx_batch - circle_yx)**2, dim=0)
    ret = torch.zeros(coordinates_yx_batch.shape[1])
    ret[dists <= circle_r**2] = 1.0
    return ret

def heatmap_batch_rectangle(coordinates_yx_batch, rect_y, rect_x, rect_h, rect_w, rect_angle):
    assert(len(coordinates_yx_batch.shape) == 2)
    assert(coordinates_yx_batch.shape[0] == 2)
    dev = coordinates_yx_batch.device
    rect_yx = torch.tensor([rect_y, rect_x]).unsqueeze(-1).to(dev)
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
    assert(len(obs) > 0)
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

def intersect_line_segment(ry, rx, dy, dx, p1y, p1x, p2y, p2x):
    """
        ry, rx   : ray (y,x) origin
        dy, dx   : ray (y,x) direction (expected to be a unit vector)
        p1y, p1x : point (y,x) for first end of line segment
        p2y, p2x : point (y,x) for second end of line segment
    """
    eps = 1e-6

    v1y = ry - p1y
    v1x = rx - p1x
    v2x = p2x - p1x
    v2y = p2y - p1y
    v3x = -dy
    v3y = dx

    v2_dot_v3 = (v2x * v3x) + (v2y * v3y)
    if abs(v2_dot_v3) < eps:
        return None
    

    v1_dot_v3 = (v1x * v3x) + (v1y * v3y)
    t2 = v1_dot_v3 / v2_dot_v3
    if t2 < 0.0 or t2 > 1.0:
        return None

    v2_cross_v1 = v2x * v1y - v2y * v1x
    t1 = v2_cross_v1 / v2_dot_v3
    return t1

def intersect_circle(ry, rx, dy, dx, sy, sx, sr):
    """
        ry, rx : ray (y,x) origin
        dy, dx : ray (y,x) direction (expected to be a unit vector)
        sy, sx : shape y,x e.g. center of circle
        sr     : shape radius
    """
    a = dy**2 + dx**2
    b = 2.0 * (dy * (ry - sy) + dx * (rx - sx))
    c = (ry - sy)**2 + (rx - sx)**2 - sr**2

    d = b**2 - 4.0 * a * c

    if d < 0.0:
        return None

    sqrtd = math.sqrt(d)

    t0 = (-b - sqrtd) / (2.0 * a)
    t1 = (-b + sqrtd) / (2.0 * a)

    e = 1e-6

    if t0 > e:
        if t1 > e:
            return min(t0, t1)
        return t0
    else:
        if t1 > e:
            return t1
        return None

def intersect_rectangle(ry, rx, dy, dx, sy, sx, sh, sw, sa):
    """
        ry, rx : ray (y,x) origin
        dy, dx : ray (y,x) direction (expected to be a unit vector)
        sy, sx : shape (y,x) e.g. center of rectangle
        sh, sw : shape height and width
        sa     : shape angle
    """
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
    
    t_vals = [
        intersect_line_segment(ry, rx, dy, dx, tl[0], tl[1], tr[0], tr[1]),
        intersect_line_segment(ry, rx, dy, dx, tr[0], tr[1], br[0], br[1]),
        intersect_line_segment(ry, rx, dy, dx, br[0], br[1], bl[0], bl[1]),
        intersect_line_segment(ry, rx, dy, dx, bl[0], bl[1], tl[0], tl[1]),
    ]

    isSomething = lambda x : x is not None

    t_vals = list(filter(isSomething, t_vals))

    if len(t_vals) == 0:
        return None

    return min(t_vals)

def distance_along_line_of_sight(obs, ry, rx, dy, dx, no_collision_val=None):
    """
        obs    : list of obstacles
        ry, rx : (y,x) ray origin
        dy, dx : (y,x) ray direction
    """
    ray = (ry, rx, dy, dx)
    
    def intersect_shape(shape_params):
        nonlocal ray
        ty = shape_params[0]
        rest = shape_params[1:]
        if ty == CIRCLE:
            assert(len(rest) == 3)
            return intersect_circle(*ray, *rest)
        elif ty == RECTANGLE:
            assert(len(rest) == 5)
            return intersect_rectangle(*ray, *rest)
        else:
            raise Exception("Unrecognized shape")
    
    isSomething = lambda x : x is not None

    collisions = list(filter(isSomething, map(intersect_shape, obs)))

    if len(collisions) == 0:
        return no_collision_val
    return min(collisions)

def line_of_sight_from_bottom_up(obs, position):
    return distance_along_line_of_sight(obs, 1.0, position, -1.0, 0.0, 1.0)

def make_image_pred(example, img_size, network, num_splits, predict_variance):
    num_splits = max(num_splits, 1)
    num_dims = 2 if predict_variance else 1
    outputs = torch.zeros(num_dims, img_size**2)
    assert(img_size**2 % num_splits) == 0
    split_size = img_size**2 // num_splits
    xy_locations = all_yx_locations(img_size).permute(1, 0).unsqueeze(0)
    d = DeviceDict({
        "input": example["input"][:1],
        
    })
    for i in range(num_splits):
        begin = i * split_size
        end = (i + 1) * split_size
        d["params"] = xy_locations[:, begin:end]
        pred = network(d)["output"]
        assert(pred.shape == (1, num_dims, split_size))
        outputs[:, begin:end] = pred.reshape(num_dims, split_size).detach()
        pred = None
    return outputs.reshape(num_dims, img_size, img_size).detach().cpu()

def make_sdf_image_pred(example, img_size, network, num_splits, predict_variance):
    return make_image_pred(example, img_size, network, num_splits, predict_variance)

def make_sdf_image_gt(example, img_size):
    obs = example["obstacles_list"][0]

    coordinates_yx = all_yx_locations(img_size)

    return sdf_batch(coordinates_yx, obs).reshape(img_size, img_size)

def red_white_blue_banded(img):
    assert(len(img.shape) == 2)
    img = img * 5.0
    r = torch.clamp(img + 1.0, min=0.0, max=1.0)
    g = torch.clamp(1.0 - torch.abs(img), min=0.0, max=1.0)
    b = torch.clamp(-img + 1.0, min=0.0, max=1.0)
    x = 0.5 + img * 5.0
    x = torch.clamp(1.0 * 5.0 * torch.abs(1.0 - 2.0 * (x - torch.floor(x))), min=0.0, max=1.0)
    rgb = torch.stack(
        (r, g, b),
        dim=2
    ) * (0.5 + 0.5 * x.unsqueeze(-1))
    z = torch.clamp(1.0 - 50.0 * torch.abs(img), min=0.0, max=1.0).unsqueeze(-1)
    z_color = torch.tensor([0.0, 0.5, 0.0]).unsqueeze(0).unsqueeze(0).to(img.device)
    rgb = rgb + (z_color - rgb) * z
    return rgb

def red_white_blue(img):
    assert(len(img.shape) == 2)
    img = img.unsqueeze(-1)
    close_to_one  = torch.clamp((2.0 * img - 1.0), 0.0, 1.0)
    close_to_half = torch.clamp((1.0 - 2.0 * torch.abs(img - 0.5)), 0.0, 1.0)
    close_to_zero = torch.clamp((1.0 - 2.0 * img), 0.0, 1.0)
    red   = torch.tensor([0.71, 0.01, 0.15]).reshape(1, 1, 3)
    white = torch.tensor([1.00, 1.00, 1.00]).reshape(1, 1, 3)
    blue  = torch.tensor([0.23, 0.30, 0.75]).reshape(1, 1, 3)
    return close_to_one * red + close_to_half * white + close_to_zero * blue
    

def make_heatmap_image_pred(example, img_size, network, num_splits, predict_variance):
    return make_image_pred(example, img_size, network, num_splits, predict_variance)

def make_heatmap_image_gt(example, img_size):
    obs = example["obstacles_list"][0]

    coordinates_yx = all_yx_locations(img_size)

    # return heatmap_batch(coordinates_yx, obs).reshape(img_size, img_size)
    return torch.clamp(img_size * -sdf_batch(coordinates_yx, obs), 0.0, 1.0).reshape(img_size, img_size)

def make_depthmap_gt(example, img_size):
    obs = example["obstacles_list"][0]
    arr = torch.zeros((img_size))
    for i, x in enumerate(np.linspace(0.0, 1.0, img_size)):
        arr[i] = line_of_sight_from_bottom_up(obs, x)
    return arr

def make_depthmap_pred(example, img_size, network):
    locations = torch.linspace(0, 1, img_size).reshape(1, img_size, 1).to(the_device)
    d = DeviceDict({
        "input": example["input"][:1],
        "params": locations
    })
    pred = network(d)["output"]
    assert(len(pred.shape) == 3)
    assert(pred.shape[0] == 1)
    outputDims = pred.shape[1]
    assert(pred.shape[2] == img_size)
    return pred.reshape(outputDims, img_size).detach().cpu()

def center_and_undelay_signal(echo_signal, y, x):
    """Shifts the input signal in time so that the expected time of wave arrival at the given location is always at the start of the signal"""
    assert(echo_signal.shape == (4, 4096))

    dist = math.hypot(y - 0.5, x - 0.5) + 0.05
    center = int(dist * dataset_field_size * magic_speed_of_sound * 4096)
    center = min(4096, center)

    def impl(channel): #, receiver_y, receiver_x):
        assert(channel.shape == (4096,))
        # dist = math.hypot(y - receiver_y, x - receiver_x)
        # center = int(dist * dataset_field_size * magic_speed_of_sound * 4096)
        # center = min(4096, center)
        shifted = torch.cat((
            channel[center:],
            torch.zeros(center)
        ), dim=0)
        assert(shifted.shape == (4096,))
        return shifted
    out = torch.stack((
        impl(echo_signal[0]), #*receiver_locations[0]),
        impl(echo_signal[1]), #*receiver_locations[1]),
        impl(echo_signal[2]), #*receiver_locations[2]),
        impl(echo_signal[3]), #*receiver_locations[3])
    ), dim=0)
    assert(out.shape == (4, 4096))
    return out


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
    
def make_implicit_params_validation(img_size, representation):
    ls = torch.linspace(0, 1, img_size)
    if (representation == "depthmap"):
        return ls.reshape(1, img_size, 1)
    return torch.stack(
        torch.meshgrid(ls, ls),
        dim=2
    ).to(the_device).reshape(1, img_size**2, 2)

def make_implicit_outputs(obs, params, representation):
    assert(len(params.shape) == 2)
    if representation == "sdf":            
        assert(params.shape[1] == 2)
        return sdf_batch(params.permute(1, 0), obs)
    elif representation == "heatmap":
        assert(params.shape[1] == 2)
        return heatmap_batch(params.permute(1, 0), obs)
    elif representation == "depthmap":   
        assert(params.shape[1] == 1) 
        num = params.shape[0]
        output = torch.zeros(num)
        # CPU implementation for now :(
        for i in range(num):
            p = params[i]
            v = line_of_sight_from_bottom_up(obs, p[0])
            output[i] = torch.tensor(v)
        return output
    else:
        raise Exception("Unrecognized representation")

def make_dense_outputs(obs, representation, img_size):
    theDict = DeviceDict({ "obstacles_list": [obs]})
    if representation == "sdf":
        output = make_sdf_image_gt(theDict, img_size)
        assert(output.shape == (img_size, img_size))
    elif representation == "heatmap":
        output = make_heatmap_image_gt(theDict, img_size)
        assert(output.shape == (img_size, img_size))
    elif representation == "depthmap":
        output = make_depthmap_gt(theDict, img_size)
        assert(len(output.shape) == 1)
        assert(output.shape[0] == img_size)
    else:
        raise Exception("Unown representation")
    return output

def make_deterministic_validation_batches_implicit(example, representation, img_size, num_splits):
    num_splits = max(num_splits, 1)
    params = make_implicit_params_validation(img_size, representation)
    obsobs = example["obstacles_list"]

    data_size = img_size if (representation == "depthmap") else img_size**2
    assert(data_size % num_splits == 0)
    split_size = data_size // num_splits

    batches = []
    for i in range(num_splits):
        begin = i * split_size
        end = (i + 1) * split_size
        output = torch.zeros(len(obsobs), split_size)
        for j, obs in enumerate(obsobs):
            output[j] = make_implicit_outputs(obs, params[0, begin:end], representation)
        d = DeviceDict(example.copy())
        d["params"] = params[:, begin:end].repeat(len(obsobs), 1, 1)
        d["output"] = output
        batches.append(d)
    return batches