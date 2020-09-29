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

def shortest_distance_to_obstacles(obs, y, x):
    def min_dist(o):
        t, oy, ox = o[:3]
        rest = o[3:]
        if (t == CIRCLE):
            r = rest[0]
            return max(0.0, math.hypot(y - oy, x - ox) - r)
        elif (t == RECTANGLE):
            rh, rw = rest
            rh *= 0.5
            rw *= 0.5
            dy = max(0.0, max((oy - rh) - y, y - (oy + rh)))
            dx = max(0.0, max((ox - rw) - x, x - (ox + rw)))
            return math.hypot(dy, dx)
        else:
            assert(False)
    return min(map(min_dist, obs))

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

def make_sdf_image_pred(example, img_size, network):
    xy_locations = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, img_size),
        torch.linspace(0, 1, img_size)
    ), dim=2).cuda().reshape(1, img_size**2, 2)
    example['params'] = xy_locations
    example['input'] = example['input'][:1]
    pred = network(example)['output']
    assert(pred.shape == (1, img_size**2))
    return pred.reshape(img_size, img_size).detach().cpu().numpy()

def make_sdf_image_gt(example, img_size):
    obs = example['obstacles_list']
    arr = torch.zeros((img_size, img_size))
    for i, y in enumerate(np.linspace(0, 1, img_size)):
        for j, x in enumerate(np.linspace(0, 1, img_size)):
            arr[i,j] = shortest_distance_to_obstacles(obs, y, x)
    return arr
    
def make_heatmap_image_pred(example, img_size, network):
    return make_sdf_image_pred(example, img_size, network)

def make_heatmap_image_gt(example, img_size):
    arr = make_sdf_image_gt(example, img_size)
    arr = 1.0 - torch.clamp(arr * img_size, 0.0, 1.0) # create some blurring/anti-aliasing right on obstacle boundaries
    return arr

def make_depthmap_gt(example, img_size):
    obs = example['obstacles_list']
    arr = torch.zeros((img_size))
    for i, x in enumerate(np.linspace(0.0, 1.0, img_size)):
        arr[i] = line_of_sight_from_bottom_up(obs, x)
    return arr

def make_depthmap_pred(example, img_size, network):
    locations = torch.linspace(0, 1, img_size).reshape(1, img_size, 1).cuda()
    example['params'] = locations
    example['input'] = example['input'][:1]
    pred = network(example)['output']
    assert(pred.shape == (1, img_size))
    return pred.reshape(img_size).detach().cpu().numpy()

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
