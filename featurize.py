import torch
import math
import scipy.interpolate
import numpy as np

from field import CIRCLE, RECTANGLE

# TODO:
# - spectrogram
# - ?

magic_speed_of_sound = 1.0 / 1050.0

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
    node_locations = []
    node_values = []
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
        node_locations.append(min_dist)
        node_values.append(effective_size)
        # heatmap += make_gaussian_1d(0.1 * effective_size, min_dist, 3.0 * effective_size, heatmap_length)

    rbfi = scipy.interpolate.Rbf(
        node_locations,
        node_values,
        function="gaussian",
        epsilon=0.02,
        smooth=0.2,
        mode="1-D"
    )
    heatmap = np.clip(rbfi(np.linspace(0.0, 1.0, heatmap_length)), 0.0, 1.0)
    return torch.tensor(heatmap).float()

        

def normalize_amplitude(waveform):
    max_amp = torch.max(torch.abs(waveform))
    if max_amp > 1e-3:
        waveform = (0.5 / max_amp) * waveform
    return waveform