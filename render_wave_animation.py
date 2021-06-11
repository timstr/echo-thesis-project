import sys
import torch
import matplotlib.animation
import matplotlib.pyplot as plt

from wave_field import Field, make_random_field
from progress_bar import progress_bar

from config_constants import wavesim_emitter_locations, wavesim_receiver_locations

size = 512


def render_animation(
    field,
    duration,
    fps=30,
    output_path="animation.mp4",
    on_render_frame=None,
    render_interval=4,
):
    plt.axis("off")
    fig = plt.figure(figsize=(8, 8), dpi=64)

    im_field = plt.imshow(field.to_image().cpu())

    eys = []
    exs = []
    for (ey, ex) in wavesim_emitter_locations:
        eys.append(ey)
        exs.append(ex)

    plt.scatter(exs, eys, s=50, c="yellow")

    rys = []
    rxs = []
    for (ry, rx) in wavesim_receiver_locations:
        rys.append(ry)
        rxs.append(rx)

    plt.scatter(rxs, rys, s=20, c="white")

    num_frames = int(duration * fps)

    step_count = 0

    def animate(i):
        nonlocal step_count
        for _ in range(render_interval):
            if on_render_frame is not None:
                on_render_frame()
            field.step()
            step_count += 1
        im_field.set_data(field.to_image().cpu())
        plt.gca().set_title("Step {}".format(step_count))
        progress_bar(i, num_frames)

    ani = matplotlib.animation.FuncAnimation(
        fig, animate, frames=num_frames, interval=1000 / fps
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

    # field = make_random_field(size, 2) # Field with obstacles
    field = Field(size)  # empty field
    field.add_obstacles(
        [
            (
                "Rectangle",
                0.15000000000000002,
                0.15000000000000002,
                0.07071067811865475,
                0.07071067811865475,
                0.7853981633974483,
            ),
            ("Circle", 0.5333333333333333, 0.6000000000000001, 0.1),
        ]
    )

    emitter_locations = [
        (size - 8, size // 2 - 200),
        (size - 8, size // 2 - 100),
        (size - 8, size // 2),
        (size - 8, size // 2 + 100),
        (size - 8, size // 2 + 200),
    ]
    steps_per_emitter = 256

    current_frame = 0

    def on_render_frame():
        nonlocal current_frame
        if current_frame % steps_per_emitter == 0:
            e = current_frame // steps_per_emitter
            if e < len(emitter_locations):
                the_field = field.get_field()
                ey = emitter_locations[e][0]
                ex = emitter_locations[e][1]

                ls = torch.linspace(0.0, size, size, device="cuda")
                y, x = torch.meshgrid((ls, ls))

                dist_sqr = (y - ey) ** 2 + (x - ex) ** 2
                amp = -2.0 * torch.exp(-dist_sqr * 0.1)

                the_field[...] += amp

                # -30
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

    num_steps = 4096
    render_interval = 4
    fps = 30
    duration = num_steps / fps / render_interval
    render_animation(
        field,
        duration=duration,
        fps=fps,
        output_path="waves.mp4",
        on_render_frame=on_render_frame,
        render_interval=render_interval,
    )


if __name__ == "__main__":
    main()
