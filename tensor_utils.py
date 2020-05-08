import torch

def cutout_circle(barrier, y, x, rad):
    assert(len(barrier.shape) == 4)
    b, c, h, w = barrier.shape
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
    mask = torch.unsqueeze(torch.unsqueeze(distsqr < rad**2, 0), 0).repeat(1, 2, 1, 1)
    barrier[mask] = 0.0

def cutout_rectangle(barrier, y, x, height, width):
    assert(len(barrier.shape) == 4)
    b, c, h, w = barrier.shape
    y0 = int(h * (y - height/2))
    y1 = int(h * (y + height/2))
    x0 = int(w * (x - width/2))
    x1 = int(w * (x + width/2))
    barrier[:, :, y0:y1, x0:x1] = 0.0

def apply_border_fringe(barrier):
    fringe_size=8
    for i in range(fringe_size):
        f = 1 - (1 - i / fringe_size)# **4
        c = 0
        barrier[:,c,i,:] *= f
        barrier[:,c,-i,:] *= f

        barrier[:,c,:,i] *= f
        barrier[:,c,:,-i] *= f