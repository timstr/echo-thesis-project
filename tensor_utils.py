import torch

def cutout_circle(barrier, y, x, rad):
    assert(len(barrier.shape) == 2)
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

def apply_border_fringe(barrier):
    fringe_size=8
    for i in range(fringe_size):
        f = 1 - (1 - i / fringe_size)# **4
        c = 0
        barrier[:,c,i,:] *= f
        barrier[:,c,-i,:] *= f

        barrier[:,c,:,i] *= f
        barrier[:,c,:,-i] *= f

# Available modes:
# - "repeat"
# - "zero"
# - "wrap"
def pad_tensor(f, mode):
    assert(len(f.shape) == 4)
    assert(mode == "repeat" or mode == "zero" or mode == "wrap")

    def zero_or_swap(a, b):
        if (mode == "zero"):
            a[...] = 0.0
            b[...] = 0.0
        elif (mode == "wrap"):
            temp = a.clone()
            a[...] = b
            b[...] = temp

    top = f[:, :, :1, :]
    bottom = f[:, :, -1:, :]
    zero_or_swap(top, bottom)

    padded_y = torch.cat(
        (top, f, bottom),
        dim=2
    )

    left = padded_y[:, :, :, :1] * 1.0
    right = padded_y[:, :, :, -1:] * 1.0
    
    zero_or_swap(left, right)

    padded = torch.cat(
        (left, padded_y, right),
        dim=3
    )

    assert(padded.shape[0] == f.shape[0])
    assert(padded.shape[1] == f.shape[1])
    assert(padded.shape[2] == f.shape[2] + 2)
    assert(padded.shape[3] == f.shape[3] + 2)

    return padded