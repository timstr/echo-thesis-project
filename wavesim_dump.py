import math
import random
import time
import pickle
import os.path
from IPython.display import clear_output

import torch
import torch.nn as nn

import numpy as np

# %matplotlib notebook
import matplotlib.pyplot as plt
import matplotlib.animation

from IPython.display import HTML

import PIL

size = 512

transmitter_location = (size - 10, size // 2)
transmitter_width = 50
transmitter_frequency = 1.0 / 4.0 # in cycles per tick
transmitter_duration = 16 / transmitter_frequency # in ticks
transmitter_amplitude = 10.0

receiver_locations = [
    (size - 10, size // 2 - 20),
    (size - 10, size // 2 - 10),
    (size - 10, size // 2 + 10),
    (size - 10, size // 2 + 20)
]

def cutout_circle(barrier, x, y, rad):
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, size),
            torch.linspace(0, 1, size)
        ),
        dim=-1
    )
    loc = torch.Tensor([[[y, x]]])
    # diffs = torch.unsqueeze(torch.unsqueeze(grid - loc, 0), 0)
    diffs = grid - loc
    distsqr = torch.sum(diffs**2, dim=-1)
    mask = torch.unsqueeze(torch.unsqueeze(distsqr < rad**2, 0), 0).repeat(1, 2, 1, 1)
    barrier[mask] = 0

def cutout_rectangle(barrier, x, y, w, h):
    x0 = int(size * (x - w/2))
    x1 = int(size * (x + w/2))
    y0 = int(size * (y - h/2))
    y1 = int(size * (y + h/2))
    barrier[:, :, y0:y1, x0:x1] = 0

def make_default_barrier():
    return torch.ones((1, 2, size, size), requires_grad=False).cuda()

def apply_random_cutouts(barrier):
    # cutout_circle(barrier, 0.3, 0.1, 0.02)
    # cutout_circle(barrier, 0.4, 0.1, 0.03)
    # cutout_circle(barrier, 0.5, 0.1, 0.04)
    # cutout_circle(barrier, 0.6, 0.1, 0.03)
    # cutout_circle(barrier, 0.7, 0.1, 0.02)
    # cutout_circle(barrier, 0.25, 0.5, 0.04)
    # cutout_circle(barrier, 0.6, 0.4, 0.02)
    # cutout_circle(barrier, 0.4, 0.3, 0.06)
    # cutout_circle(barrier, 0.8, 0.3, 0.02)
    # cutout_rectangle(barrier, 0.75, 0.25, 0.08, 0.06)
    n = random.randint(1, 10)
    for _ in range(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 0.75)
        if random.random() < 0.5:
            # rectangle
            w = random.uniform(0.01, 0.2)
            h = random.uniform(0.01, 0.2)
            cutout_rectangle(barrier, x, y, w, h)
        else:
            # circle
            r = random.uniform(0.01, 0.1)
            cutout_circle(barrier, x, y, r)
    


barrier = make_default_barrier()

def randomize_barrier():
    global barrier
    barrier = make_default_barrier()
    apply_random_cutouts(barrier)
    apply_border_fringe(barrier)
    
randomize_barrier()
    
plt.imshow(barrier[0,0,:,:].cpu())

def make_default_field():
    return torch.zeros((1, 2, size, size), requires_grad=False).cuda()

field = make_default_field()


def reset_field():
    global field
    field = make_default_field()

# plt.imshow(field[0,0,:,:].cpu())

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

kernel = torch.Tensor([
    [pos_to_pos, vel_to_pos],
    [pos_to_vel, vel_to_vel]
])
kernel.requires_grad = False
# Padding mode can be 'constant', 'reflect', 'replicate', or 'circular'
conv = nn.Conv2d(2, 2, 3, 1, padding=1, padding_mode='constant', bias=False)
conv.weight = nn.Parameter(kernel, requires_grad=False)
conv = conv.cuda()

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(pos_to_pos)
ax[0,1].imshow(pos_to_vel)
ax[1,0].imshow(vel_to_pos)
ax[1,1].imshow(vel_to_vel)
fig.show()

current_timestep = 0

def reset_sim():
    global current_timestep
    current_timestep = 0
    randomize_barrier()
    reset_field()

def perturb_field(t):
    y, x = transmitter_location
#     w = transmitter_width
#     n = transmitter_duration
#     f = transmitter_frequency
#     a = transmitter_amplitude
#     if (t < n):
#         theta = t * 2.0 * math.pi * f
#         v = a * math.sin(theta)
#         field[0, 0, y, x-(w//2):x+(w//2)] = v
    if t == 0:
        field[0,0,y,x] = 10
    
def sample_field():
    out = []
    for y, x in receiver_locations:
        v = field[0,0,y,x].item()
        out.append(v)
    return out

def step_sim():
    global current_timestep
    global field
    perturb_field(current_timestep)
    current_timestep += 1
    field = conv(field)
    field = field * barrier

def raymarch(b, x, y, dirx, diry, fov=60, res=128, step_size=1):
    b = b.cpu()
    w = b.shape[2]
    h = b.shape[3]
    view_angle = math.atan2(diry, dirx)
    fov *= math.pi / 180
    image = torch.zeros((res))
    for i in range(res):
        t = i / (res - 1)
        ray_angle = view_angle + (fov * (0.5 - t))
        dx = step_size * math.cos(ray_angle)
        dy = step_size * math.sin(ray_angle)
        px = float(x)
        py = float(y)
        dist = 0
        while True:
            pxi = int(px)
            pyi = int(py)
            if pxi < 0 or pxi >= w or pyi < 0 or pyi >= h:
                break
            if b[0,0,pxi,pyi] < 1:
                break
            px += dx
            py += dy
            dist += step_size
        image[i] = dist
    return image

# fig, ax = plt.subplots(1,2, figsize=(8,4), dpi=80)
degrees = np.linspace(0, math.pi, 128)
depthmap = raymarch(barrier, size-10, size//2 - 10, -1, 0, fov=180, res=128, step_size=0.5)
plt.polar(degrees, depthmap)
plt.show()
plt.plot(degrees, depthmap)

def generate_echo_and_depthmap():
    clear_output()
    
    reset_sim()
    
    plt.imshow(barrier[0,0,:,:].cpu())
    plt.show()

    receiver_buf = []
    receiver_dur = size * 16 # 44100

    for _ in range(receiver_dur):
        step_sim()
        v = sample_field()
        receiver_buf.append(v)
        
    trim_amount = 0 # 512
    receiver_buf = np.array(receiver_buf)
    receiver_buf = receiver_buf[trim_amount:]

    plt.plot(receiver_buf)
    plt.show()
    
    depthmap = raymarch(
        barrier,
        transmitter_location[0],
        transmitter_location[1],
        -1,
        0,
        fov=60,
        res=128,
        step_size=0.5
    )
    plt.plot(depthmap)
    plt.show()
    
    
    
    return (
        barrier[0,0,:,:].cpu().numpy(),
        receiver_buf,
        depthmap.cpu().numpy()
    )

def create_and_store_dataset(n):
    path = "dataset/v3"
    for i in range(n):
        data = generate_echo_and_depthmap()
        fname = "{0}/example {1}.pkl".format(path, i + 528)
        assert(not os.path.isfile(fname))
        print(fname)
        with open(fname, "wb") as file:
            pickle.dump(data, file)
        
generate_echo_and_depthmap()

create_and_store_dataset(1000 - 527)

import scipy.io.wavfile as wf

sound_buf = np.asarray(receiver_buf)
sound_max_amp = np.max(np.abs(sound_buf))
if sound_max_amp > 1e-3:
    sound_buf *= 0.5 / sound_max_amp

wf.write(
    "circle and bar.wav",
    44100,
    sound_buf
)

def render_animation(duration, fps=30):
    reset_sim()
    
    plt.axis('off')
    fig = plt.figure(
        figsize=(8,8),
        dpi=64
    )
    
    im_field = plt.imshow(
        field[0,0,:,:].cpu(),
        cmap="inferno",
        vmin = -0.05,
        vmax = 0.05
    )
    barrier_rgba = torch.zeros(size, size, 4)
    barrier_rgba[:,:,3] = 1.0 - barrier[0,0,:,:].cpu()
    im_barrier = plt.imshow(
        barrier_rgba
    )

    def animate(i):
        for _ in range(4):
            step_sim()
        im_field.set_data(field[0,0,:,:].cpu())

    ani = matplotlib.animation.FuncAnimation(
        fig,
        animate,
        frames=duration*fps,
        interval = 1000/fps
    )

    return HTML(ani.to_html5_video())

render_animation(duration=60)