import sys
import torch
import matplotlib.animation
import matplotlib.pyplot as plt

from field import Field, make_random_field
from progress_bar import progress_bar

def render_animation(field, duration, fps=30, output_path="animation.mp4", on_render_frame=None):
    plt.axis('off')
    fig = plt.figure(
        figsize=(8,8),
        dpi=64
    )
    
    im_field = plt.imshow(
        field.get_field()[0,0,:,:].cpu(),
        cmap="inferno",
        vmin = -0.05,
        vmax = 0.05
    )
    barrier_rgba = torch.zeros(field.get_height(), field.get_width(), 4)
    barrier_rgba[:,:,3] = 1.0 - field.get_barrier()[0,0,:,:].cpu()
    im_barrier = plt.imshow(
        barrier_rgba
    )

    num_frames = duration*fps

    def animate(i):
        for _ in range(4):
            if on_render_frame is not None:
                on_render_frame()
            field.step()
        im_field.set_data(field.get_field()[0,0,:,:].cpu())
        progress_bar(i, num_frames)


    ani = matplotlib.animation.FuncAnimation(
        fig,
        animate,
        frames=num_frames,
        interval = 1000/fps
    )

    ani.save(output_path)

    sys.stdout.write("\nSaved animation to {}".format(output_path))

def main():
    size = 512
 
    field = make_random_field(size, size)
    # field.get_field()[0,0,size//2, size//2] = 10.0

    # field = Field(size, size)
    y = torch.linspace(-1.0, 1.0, size)
    x = 0.5 * torch.linspace(-1.0, 1.0, size)
    grid = torch.stack(
        torch.meshgrid(y, x),
        dim=-1
    )
    loc = torch.Tensor([[[0.5, 0.0]]])
    diffs = grid - loc
    distsqr = torch.sum(diffs**2, dim=-1)
    gaussian_mask = torch.exp(-distsqr * 100.0)
    freq = y * 100.0
    amp = 0.04
    
    field.get_field()[0,0,:,:] = amp * torch.sin(freq * y).unsqueeze(-1) * gaussian_mask
    field.get_field()[0,1,:,:] = 0.1 * amp * torch.cos(freq * y).unsqueeze(-1) * gaussian_mask
    
    plt.imshow(field.get_barrier()[0,0,:,:].cpu())
    plt.show()

    render_animation(field, duration=60, fps=30, output_path="waves.mp4")

if __name__ == "__main__":
    main()