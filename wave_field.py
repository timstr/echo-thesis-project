import torch

from wave_simulation import step_simulation
from shape_types import CIRCLE, RECTANGLE
from featurize import make_random_obstacles, all_yx_locations, heatmap_batch

from the_device import the_device

# TODO: test whether doing convolutions in-place is faster (e.g. using torch's `out` parameters)


class Field:
    def __init__(self, size):
        self._size = size
        self._obstacles = []
        self._dirty_barrier = True
        self._field_now = torch.zeros(size, size).to(the_device)
        self._field_prev = torch.zeros(size, size).to(the_device)

    def get_size(self):
        return self._size

    def silence(self):
        self._field_now[...] = 0.0
        self._field_prev[...] = 0.0

    def add_obstacles(self, obstacles):
        for o in obstacles:
            t = o[0]
            args = o[1:]
            if t == CIRCLE:
                self.add_circle(*args)
            elif t == RECTANGLE:
                self.add_rectangle(*args)

    def add_circle(self, y, x, radius):
        self._obstacles.append((CIRCLE, y, x, radius))
        self._dirty_barrier = True

    def add_rectangle(self, y, x, height, width, angle):
        self._obstacles.append((RECTANGLE, y, x, height, width, angle))
        self._dirty_barrier = True

    def get_obstacles(self):
        return self._obstacles

    def _update_barrier(self):
        if not self._dirty_barrier:
            return

        if len(self._obstacles) == 0:
            self._barrier = torch.tensor([[1.0]], dtype=torch.float).to(the_device)
            return

        coordinates_yx_batch = all_yx_locations(self._size)

        self._barrier = heatmap_batch(coordinates_yx_batch, self._obstacles)
        self._barrier = self._barrier.reshape(self._size, self._size).to(the_device)
        self._barrier = 1.0 - self._barrier

        self._dirty_barrier = False

    def get_barrier(self):
        if self._dirty_barrier:
            self._update_barrier()
        assert not self._dirty_barrier
        return self._barrier

    def get_field(self):
        return self._field_now

    def step(self):
        self._update_barrier()

        self._field_now, self._field_prev = step_simulation(
            self._field_now, self._field_prev
        )
        self._field_now *= self._barrier

        assert self._field_now.shape == (self._size, self._size)
        assert self._field_prev.shape == (self._size, self._size)

    def to_image(self):
        self._update_barrier()
        amp = 5.0
        # Red: positive amplitude
        r = torch.clamp(amp * self._field_now, 0.0, 1.0)
        # Green: absolute amplitude minus one
        g = torch.clamp(torch.abs(-amp * self._field_now) - 1.0, 0.0, 1.0)
        # Blue: negative amplitude
        b = torch.clamp(-amp * self._field_now, 0.0, 1.0)
        rgb = torch.stack((r, g, b), dim=2)
        # Make obstacles appear grey
        rgb += (0.5 - rgb) * (1.0 - self._barrier.unsqueeze(-1))
        return rgb


def make_random_field(size, max_num_obstacles=10):
    f = Field(size)
    f.add_obstacles(make_random_obstacles(max_num_obstacles))
    return f
