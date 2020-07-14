import sys
import torch
import matplotlib.animation
import matplotlib.pyplot as plt
import math

from field import Field, make_random_field
from progress_bar import progress_bar

def render_animation(field, duration, fps=30, output_path="animation.mp4", on_render_frame=None, render_interval=4):
    plt.axis('off')
    fig = plt.figure(
        figsize=(8,8),
        dpi=64
    )
    
    def make_field_image():
        amp = 20.0 * field.get_field()[0,0,:,:]
        vel =  5.0 * field.get_field()[0,1,:,:]
        field_rgb = torch.zeros(field.get_height(), field.get_width(), 3)
        # Red: positive amplitude
        field_rgb[:,:,0] = torch.clamp(amp, 0.0, 1.0)
        # Blue: negative amplitude
        field_rgb[:,:,2] = torch.clamp(-amp , 0.0, 1.0)
        # Green: absolute velocity
        field_rgb[:,:,1] = torch.clamp(torch.abs(vel), 0.0, 1.0)
        return field_rgb.cpu()

    im_field = plt.imshow(
        make_field_image()
        # cmap="inferno",
        # vmin = -0.05,
        # vmax = 0.05
    )
    barrier_rgba = torch.ones(field.get_height(), field.get_width(), 4) * 0.5
    barrier_rgba[:,:,3] = 1.0 - field.get_barrier()[0,0,:,:].cpu()
    im_barrier = plt.imshow(
        barrier_rgba
    )

    num_frames = int(duration * fps)

    step_count = 0

    def animate(i):
        nonlocal step_count
        for _ in range(render_interval):
            if on_render_frame is not None:
                on_render_frame()
            field.step()
            step_count += 1
        im_field.set_data(make_field_image())
        plt.gca().set_title("Step {}".format(step_count))
        progress_bar(i, num_frames)


    ani = matplotlib.animation.FuncAnimation(
        fig,
        animate,
        frames=num_frames,
        interval = 1000/fps
    )

    ani.save(output_path)

    print("Simulation ran for {} steps".format(step_count))

    sys.stdout.write("\nSaved animation to {}".format(output_path))

# def make_wavelet(field):
#     _, _, h, w = field.get_field().shape
#     y = torch.linspace(-1.0, 1.0, h)
#     x = 0.5 * torch.linspace(-1.0, 1.0, w)
#     grid = torch.stack(
#         torch.meshgrid(y, x),
#         dim=-1
#     )
#     loc = torch.Tensor([[[0.5, 0.0]]])
#     diffs = grid - loc
#     distsqr = torch.sum(diffs**2, dim=-1)
#     gaussian_mask = torch.exp(-distsqr * 100.0)
#     freq = y * 100.0
#     amp = 0.1
    
#     field.get_field()[0,0,:,:] = amp * torch.sin(freq * y).unsqueeze(-1) * gaussian_mask
#     field.get_field()[0,1,:,:] = 0.75 * amp * torch.cos(freq * y).unsqueeze(-1) * gaussian_mask

def main():
    size = 512

    field = make_random_field(size, size) # Field with obstacles
    # field = Field(size, size) # empty field

    cycle_length = 200
    num_cycles = 8
    signal_len = num_cycles * cycle_length
    signal = 2.0 * torch.round(0.5 + 0.5 * torch.sin(torch.linspace(0.0, num_cycles * 2.0 * math.pi, signal_len))) - 1.0
    
    current_frame = 0

    speaker_width = 128
    speaker_amplitude = torch.cos(torch.linspace(-0.5, 0.5, speaker_width) * math.pi)

    def on_render_frame():
        nonlocal current_frame
        if (current_frame < signal_len):
            signal_val = signal[current_frame]
            field.get_field()[
                0,
                0,
                size-8:size-4,
                (size//2 - speaker_width//2) : (size//2 + speaker_width//2)
            ] = signal_val * speaker_amplitude.unsqueeze(0)
        current_frame += 1

    # make_wavelet(field)
    # field.get_field()[0,0,32:42, 32:42] = 0.25 # single impulse
    # rad = 1
    # field.get_field()[0,0,size//2-rad:size//2+rad,size//2-rad:size//2+rad] = 2.5
    
    # plt.imshow(field.get_barrier()[0,0,:,:].cpu())
    # plt.show()

    # for i in range(600):
    #     field.step()
    # plt.imshow(field.get_field()[0,0,:,:].cpu())
    # plt.show()

    num_steps = 16384
    render_interval = 8
    fps = 30
    duration = num_steps / fps / render_interval
    render_animation(
        field,
        duration=duration,
        fps=fps,
        output_path="waves.mp4",
        on_render_frame=on_render_frame,
        render_interval=render_interval
    )

if __name__ == "__main__":
    main()