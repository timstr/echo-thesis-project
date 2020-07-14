import math
import torch

from field import Field

# TODO: this is unnecessary and inefficient.
# Simple line-rectangle and line-circle intersection tests should suffice, similar to how SDF is computed

def raymarch(field, y, x, diry, dirx, fov=60, res=128, step_size=1):
    barrier = field.get_barrier()#.cpu()
    assert(len(barrier.shape) == 4)
    b, c, h, w = barrier.shape
    view_angle = math.atan2(diry, dirx)
    fov *= math.pi / 180

    positions = torch.tensor([[float(y)], [float(x)]]).repeat((1, res)).cuda()
    angles = torch.linspace(-0.5, 0.5, res) * fov
    directions = torch.stack([
        torch.sin(angles),
        torch.cos(angles)
    ], dim=1).permute(1, 0).cuda()
    distances = torch.zeros((res))

    for i in range(int(math.ceil(math.hypot(h, w) / step_size))):
        coords = positions.round().to(torch.long)
        ycoords = coords[0,:]
        xcoords = coords[1,:]
        y_in_bounds = (ycoords >= 0) * (ycoords < h)
        x_in_bounds = (xcoords >= 0) * (xcoords < w)
        mask = y_in_bounds * x_in_bounds

        mask[mask] = barrier[0,0,ycoords[mask],xcoords[mask]] >= 1.0

        mask_2d = mask.unsqueeze(0).repeat(2, 1)
        positions[mask_2d] += step_size * directions[mask_2d]
        distances[mask] += step_size
    return distances