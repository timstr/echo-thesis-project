import torch
import math
import scipy.interpolate
import numpy as np

from field import CIRCLE, RECTANGLE
from device_dict import DeviceDict

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

# creates a Gaussian bell curve in the domain x \in [0, 1]
def make_gaussian_1d(width, center, amplitude, length):
    x = torch.linspace(0.0, 1.0, steps=length)
    return amplitude * torch.exp(-((x - center) / width)**2)

# TODO: remove hardcoded width and speed of sound
def make_first_echo_bump(mindist, length):
    width = 0.01
    mindist = mindist * magic_speed_of_sound
    return make_gaussian_1d(width, mindist, 1.0, length)

def make_obstacle_heatmap(obstacles, field_size, heatmap_length):
    center_y = 0.5
    center_x = 0.5
    # node_locations = []
    # node_values = []
    heatmap = torch.zeros(heatmap_length)
    for obs in obstacles:
        kind = obs[0]
        obs_y = obs[1]
        obs_x = obs[2]
        shape_args = obs[3:]
        if (kind == CIRCLE):
            radius = shape_args[0]
            min_dist = max(0.0, math.hypot(obs_y - center_y, obs_x - center_x) - radius)
            effective_size = 2 * radius
        elif (kind == RECTANGLE):
            rect_h = shape_args[0]
            rect_w = shape_args[1]
            top = obs_y - (rect_h / 2)
            bottom = obs_y + (rect_h / 2)
            left = obs_x - (rect_w / 2)
            right = obs_x + (rect_w / 2)
            min_dist = max(0.0, min([
                math.hypot(top - center_y, left - center_x),
                math.hypot(top - center_y, right - center_x),
                math.hypot(bottom - center_y, left - center_x),
                math.hypot(bottom - center_y, right - center_x),
            ]))
            effective_size = math.hypot(rect_h, rect_w) / min_dist
        
        min_dist = min_dist * field_size * magic_speed_of_sound
        # node_locations.append(min_dist)
        # node_values.append(effective_size)
        heatmap += make_gaussian_1d(0.01, min_dist, 3.0 * effective_size, heatmap_length)

    # rbfi = scipy.interpolate.Rbf(
    #     node_locations,
    #     node_values,
    #     function="gaussian",
    #     epsilon=0.02,
    #     smooth=0.2,
    #     mode="1-D"
    # )
    # heatmap = np.clip(rbfi(np.linspace(0.0, 1.0, heatmap_length)), 0.0, 1.0)
    # return torch.tensor(heatmap).float()
    return torch.clamp(heatmap, 0.0, 1.0)

        

def normalize_amplitude(waveform):
    max_amp = torch.max(torch.abs(waveform))
    if max_amp > 1e-3:
        waveform = (0.5 / max_amp) * waveform
    return waveform


def rotate_obstacle_90_deg(o):
    t, y, x = o[:3]
    rest = o[3:]
    if (t == RECTANGLE):
        h, w = rest
        rest = w, h
    elif (t == CIRCLE):
        assert(len(rest) == 1)
        # nothing needs to be done with radius
    return (t, 1.0 - x, y) + rest

def rotate_example_90_deg(obs, echo):
    obs = list(map(rotate_obstacle_90_deg, obs))
    assert(echo.shape == (4, 4096))
    echo = torch.stack(
        (echo[2,:], echo[0,:], echo[3,:], echo[1,:]),
        dim=0
    )
    assert(echo.shape == (4, 4096))
    return obs, echo

def rotate_example_90_deg_n_times(obs, echo, n):
    assert(type(n) == int)
    assert(n >= 0 and n < 4)
    if (n == 0):
        return obs, echo
    return rotate_example_90_deg_n_times(
        *rotate_example_90_deg(obs, echo),
        n - 1
    )

def flip_obstacle_over_x_axis(o):
    return o[:2] + (1.0 - o[2],) + o[3:]

def flip_example_over_x_axis(obs, echo):
    obs = list(map(flip_obstacle_over_x_axis, obs))
    assert(echo.shape == (4, 4096))
    echo = torch.stack(
        (echo[1,:], echo[0,:], echo[3,:], echo[2,:]),
        dim=0
    )
    assert(echo.shape == (4, 4096))
    return obs, echo


def permute_example(obs, echo, index):
    if index >= 4:
        obs, echo = flip_example_over_x_axis(obs, echo)
        index -= 4
    return rotate_example_90_deg_n_times(obs, echo, index)

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

# def make_sdf_test_example(input_example, y, x):
#     """Given an example from the dataset, produces a new example (with batch size 1) for evaluating the SDF at the given location"""
#     obs = input_example['obstacles_list']
#     echo_raw = input_example['echo_raw'][0]
#     echo_waveshaped = sclog(torch.tensor(echo_raw))
#     echo_waveshaped = add_signal_heatmap(echo_waveshaped, y, x).to(echo_raw.device)
#     sdf_yx = torch.tensor([y, x]).to(echo_raw.device)
#     sdf_value = torch.tensor([
#         shortest_distance_to_obstacles(obs, y, x)
#     ]).to(echo_raw.device)

#     return DeviceDict({
#         'obstacles_list':   input_example['obstacles_list'][:1],
#         'obstacles':        input_example['obstacles'][:1],
#         'echo_raw':         echo_raw.unsqueeze(0),
#         'echo_waveshaped':  echo_waveshaped.unsqueeze(0),
#         'depthmap':         input_example['depthmap'][:1],
#         'heatmap':          input_example['heatmap'][:1],
#         'sdf_location':     sdf_yx.unsqueeze(0),
#         'sdf_value':        sdf_value.unsqueeze(0)
#     })

def make_sdf_image_pred(example, sdf_img_size, network):
    # img = np.zeros((sdf_img_size, sdf_img_size))
    # for i, y in enumerate(np.linspace(0, 1, sdf_img_size)):
    #     for j, x in enumerate(np.linspace(0, 1, sdf_img_size)):
    #         ex = make_sdf_test_example(example, y, x)
    #         pred = network(ex)['sdf_value'][0]
    #         img[i,j] = pred.item()
    # return img

    xy_locations = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, sdf_img_size),
        torch.linspace(0, 1, sdf_img_size)
    ), dim=2).cuda().reshape(sdf_img_size**2, 2)
    assert(xy_locations.shape == (sdf_img_size**2, 2))
    example['sdf_location'] = xy_locations
    example['echo_waveshaped'] = example['echo_waveshaped'][:1].expand(sdf_img_size**2, -1, -1)
    pred = network(example)['sdf_value']
    assert(pred.shape == (sdf_img_size**2, 1))
    return pred.reshape(sdf_img_size, sdf_img_size).detach().cpu().numpy()

def make_sdf_image_gt(example, sdf_img_size):
    obs = example['obstacles_list']
    img = np.zeros((sdf_img_size, sdf_img_size))
    for i, y in enumerate(np.linspace(0, 1, sdf_img_size)):
        for j, x in enumerate(np.linspace(0, 1, sdf_img_size)):
            img[i,j] = shortest_distance_to_obstacles(obs, y, x)
    return img

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

# def add_signal_heatmap(echo_signal, y, x):
#     assert(echo_signal.shape == (4, 4096))
#     dist = math.hypot(y - 0.5, x - 0.5)
#     center = dist * dataset_field_size * magic_speed_of_sound
#     hm = make_gaussian_1d(0.1, center, 1.0, 4096).to(echo_signal.device)
#     out = torch.cat((
#         hm.unsqueeze(0),
#         echo_signal
#     ), dim=0)
#     assert(out.shape == (5, 4096))
#     return out
