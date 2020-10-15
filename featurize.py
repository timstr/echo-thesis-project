import torch
import math
import scipy.interpolate
import numpy as np
import random

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

# signed, clipped logarithm
def sclog(t):
    max_val = 1e0
    min_val = 1e-4
    signs = torch.sign(t)
    t = torch.abs(t)
    t = torch.clamp(t, min=min_val, max=max_val)
    t = torch.log(t)
    t = (t - math.log(min_val)) / (math.log(max_val) - math.log(min_val))
    t = t * signs
    return t

def normalize_amplitude(waveform):
    max_amp = torch.max(torch.abs(waveform))
    if max_amp > 1e-3:
        waveform = (0.5 / max_amp) * waveform
    return waveform

def sdf_batch_circle(coordinates_yx_batch, circle_y, circle_x, circle_r):
    assert(len(coordinates_yx_batch.shape) == 2)
    assert(coordinates_yx_batch.shape[0] == 2)
    dev = coordinates_yx_batch.device
    circle_yx = torch.tensor([circle_y, circle_x]).unsqueeze(-1).to(dev)
    return torch.sqrt(torch.sum((coordinates_yx_batch - circle_yx)**2, dim=0)) - circle_r

def sdf_batch_rectangle(coordinates_yx_batch, rect_y, rect_x, rect_h, rect_w):
    assert(len(coordinates_yx_batch.shape) == 2)
    assert(coordinates_yx_batch.shape[0] == 2)
    dev = coordinates_yx_batch.device
    rect_yx = torch.tensor([rect_y, rect_x]).unsqueeze(-1).to(dev)
    rect_hw_halved = torch.tensor([rect_h * 0.5, rect_w * 0.5]).unsqueeze(-1).to(dev)
    edgeDisp = torch.abs(coordinates_yx_batch - rect_yx) - rect_hw_halved
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
        h, w = params
        return sdf_batch_rectangle(coordinates_yx_batch, y, x, h, w)
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

def heatmap_batch_rectangle(coordinates_yx_batch, rect_y, rect_x, rect_h, rect_w):
    assert(len(coordinates_yx_batch.shape) == 2)
    assert(coordinates_yx_batch.shape[0] == 2)
    dev = coordinates_yx_batch.device
    rect_yx = torch.tensor([rect_y, rect_x]).unsqueeze(-1).to(dev)
    abs_disps = torch.abs(coordinates_yx_batch - rect_yx)
    ret = torch.zeros(coordinates_yx_batch.shape[1])
    inside_y = abs_disps[0] < (rect_h * 0.5)
    inside_x = abs_disps[1] < (rect_w * 0.5)
    ret[inside_y * inside_x] = 1.0
    return ret

def heatmap_batch_one_shape(coordinates_yx_batch, shape_tuple):
    ty, y, x = shape_tuple[:3]
    params = shape_tuple[3:]
    if ty == CIRCLE:
        r, = params
        return heatmap_batch_circle(coordinates_yx_batch, y, x, r)
    elif ty == RECTANGLE:
        h, w = params
        return heatmap_batch_rectangle(coordinates_yx_batch, y, x, h, w)
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
    e = 1e-6

    # v1 : vector from first line end to ray origin
    v1y = ry - p1y
    v1x = rx - p1x
    # v2 : vector along line segment
    v2y = p2y - p1y
    v2x = p2x - p1x

    # v3: vector orthogonal to ray direction
    v3y = dx
    v3x = -dy

    v1dotv3 = v1y * v3y + v1x * v3x
    v2dotv3 = v2y * v3y + v2x * v3x
    if abs(v2dotv3) < e:
        return None

    # position of collision point along line segment from p1 to p2
    t2 = v1dotv3 / v2dotv3

    if t2 < 0.0 or t2 > 1.0:
        return None

    v2crossv1 = math.hypot(v2y * v1x, v2x * v1y)

    # position of collision point along ray
    t1 = v2crossv1 / v2dotv3

    return t1 if t1 > e else None


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

def intersect_rectangle(ry, rx, dy, dx, sy, sx, sh, sw):
    """
        ry, rx : ray (y,x) origin
        dy, dx : ray (y,x) direction (expected to be a unit vector)
        sy, sx : shape y,x e.g. center of rectangle
        sh, sw : shape height and width
    """
    top = sy - (0.5 * sh)
    bottom = sy + (0.5 * sh)
    left = sx - (0.5 * sw)
    right = sx + (0.5 * sw)
    t_vals = [
        intersect_line_segment(ry, rx, dy, dx, top,    left,  top,    right),
        intersect_line_segment(ry, rx, dy, dx, bottom, left,  bottom, right),
        intersect_line_segment(ry, rx, dy, dx, top,    left,  bottom, left),
        intersect_line_segment(ry, rx, dy, dx, top,    right, bottom, right)
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
            assert(len(rest) == 4)
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
    xy_locations = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, img_size),
        torch.linspace(0, 1, img_size)
    ), dim=2).cuda().reshape(1, img_size**2, 2)
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

    ls = torch.linspace(0, 1, img_size)
    coordinates_yx = torch.stack(
        torch.meshgrid(ls, ls),
        dim=2
    ).reshape(img_size**2, 2).permute(1, 0).cuda()

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

    ls = torch.linspace(0, 1, img_size)
    coordinates_yx = torch.stack(
        torch.meshgrid(ls, ls),
        dim=2
    ).reshape(img_size**2, 2).permute(1, 0).cuda()

    # return heatmap_batch(coordinates_yx, obs).reshape(img_size, img_size)
    return torch.clamp(img_size * -sdf_batch(coordinates_yx, obs), 0.0, 1.0).reshape(img_size, img_size)

def make_depthmap_gt(example, img_size):
    obs = example["obstacles_list"][0]
    arr = torch.zeros((img_size))
    for i, x in enumerate(np.linspace(0.0, 1.0, img_size)):
        arr[i] = line_of_sight_from_bottom_up(obs, x)
    return arr

def make_depthmap_pred(example, img_size, network):
    locations = torch.linspace(0, 1, img_size).reshape(1, img_size, 1).cuda()
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

def make_receiver_indices(receivers, arrangement):
    assert(isinstance(receivers, int))
    assert(arrangement == "flat" or arrangement == "grid")

    # NOTE: see "receiver locations.txt" to understand what's happening here

    assert(receivers >= 4 or arrangement == "flat")
    assert(receivers <= 16 or arrangement == "grid")

    if arrangement == "grid" and receivers >= 4:
        if receivers >= 16:
            numrows = 4
        else:
            numrows = 2
    else:
        numrows = 1

    numcols = receivers // numrows

    receiver_indices = []
    for i in range(numcols):
        if numcols == 1:
            idx_base = 32
        elif numcols == 2:
            idx_base = 16 + 32 * i
        else:
            idx_base = ((i * 64) // numcols) + ((16 // (numcols * 2)) & 0xFFF8)
        for j in range(numrows):
            receiver_indices.append(idx_base + (j * 3 if numrows == 2 else j))
    
    assert(len(receiver_indices) == receivers)

    return receiver_indices

def overlapping(obstacle1, obstacle2):
    t1 = obstacle1[0]
    t2 = obstacle2[0]
    assert(t1 == CIRCLE or t1 == RECTANGLE)
    assert(t2 == CIRCLE or t2 == RECTANGLE)
    args1 = obstacle1[1:]
    args2 = obstacle2[1:]
    if t1 == CIRCLE and t2 == CIRCLE:
        y1, x1, r1 = args1
        y2, x2, r2 = args2
        d = math.hypot(y1 - y2, x1 - x2)
        return d < r1 + r2
    elif t1 == RECTANGLE and t2 == RECTANGLE:
        y1, x1, h1, w1 = args1
        y2, x2, h2, w2 = args2
        return abs(y1 - y2) < (h1 + h2) / 2 and abs(x1 - x2) < (w1 + w2) / 2
    else:
        ry, rx, rw, rh = args1 if t1 == RECTANGLE else args2
        cy, cx, cr = args1 if t1 == CIRCLE else args2
        return abs(ry - cy) < rh / 2 + cr and abs(rx - cx) < rw / 2 + cr

def make_random_obstacles(max_num_obstacles=10):
    n = random.randint(1, max_num_obstacles)

    obs = []

    def collision(shape):
        nonlocal obs
        for s in obs:
            if (overlapping(s, shape)):
                return True
        return False

    obs = []

    for _ in range(n):
        for _ in range(100):
            y = random.uniform(0, 0.75)
            # y = random.uniform(0, 1)
            x = random.uniform(0, 1)
            if random.random() < 0.5:
                # rectangle
                h = random.uniform(0.01, 0.2)
                w = random.uniform(0.01, 0.2)
                o = (RECTANGLE, y, x, h, w)
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
    ).cuda().reshape(1, img_size**2, 2)

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
            v = torline_of_sight_from_bottom_up(obs, p[0])
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
    return output

def make_deterministic_validation_batches_implicit(example, representation, img_size, num_splits):
    num_splits = max(num_splits, 1)
    params = make_implicit_params_validation(img_size, representation)
    obsobs = example["obstacles_list"]

    assert(img_size**2 % num_splits == 0)
    split_size = img_size**2 // num_splits

    batches = []
    for i in range(num_splits):
        output = torch.zeros(len(obsobs), split_size)
        for j, obs in enumerate(obsobs):
            begin = i * split_size
            end = (i + 1) * split_size
            output[j] = make_implicit_outputs(obs, params[0, begin:end], representation)
        d = DeviceDict(example.copy())
        d["params"] = params[:, begin:end].repeat(len(obsobs), 1, 1)
        d["output"] = output
        batches.append(d)
    return batches