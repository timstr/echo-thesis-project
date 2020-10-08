import torch
import random
import math

from tensor_utils import cutout_circle, cutout_rectangle
from wave_kernel import make_wave_kernel, pad_field
from featurize import CIRCLE, RECTANGLE, make_random_obstacles, overlapping, make_obstacle_heatmap

# TODO: test whether doing convolutions in-place is faster (e.g. using torch's `out` parameters)

class Field():
    def __init__(self, size):
        self._size = size
        self._obstacles = []
        self._dirty_barrier = True
        self._wave_kernel = make_wave_kernel(
            propagation_speed=1.0,
            time_step=0.1,
            velocity_damping=0.999,
            velocity_dispersion=0.05
        )
        self._padding_parameters = torch.Tensor([
            0.2731468678,  0.3291435242, -0.5704568028, -0.1916972548, -0.0072381911, -0.0069156736, -0.0045501599,  0.0087358886,
            -0.0653517470,  0.1985650659, -0.3193856180,  0.1051328108, -0.0030024105, -0.0037237250, -0.0020352036,  0.0133369453,
            0.2357937992,  0.2602138519, -0.5818488598, -0.2399103791, -0.0069989869, -0.0064358339, -0.0043960437,  0.0090550631,
            -0.1118448377,  0.1101172492, -0.3644709289,  0.0389745384, -0.0027823041, -0.0032603526, -0.0019526740,  0.0135646062
        ]).cuda()
        self._field = torch.zeros(
            (1, 2, size, size),
            requires_grad=False,
            dtype=torch.float32
        ).cuda()

    def get_size(self):
        return self._size

    def reset_field():
        self._field[...] = 0.0

    def add_obstacles(self, obstacles):
        for o in obstacles:
            t = o[0]
            args = o[1:]
            n_args = len(args)
            if (t == CIRCLE):
                assert(n_args == 3)
                self.add_circle(args[0], args[1], args[2])
            elif (t == RECTANGLE):
                assert(n_args == 4)
                self.add_rectangle(args[0], args[1], args[2], args[3])

    def add_circle(self, y, x, radius):
        self._obstacles.append((CIRCLE, y, x, radius))
        self._dirty_barrier = True

    def add_rectangle(self, y, x, height, width):
        self._obstacles.append((RECTANGLE, y, x, height, width))
        self._dirty_barrier = True

    def get_obstacles(self):
        return self._obstacles

    def _update_barrier(self):
        if not self._dirty_barrier:
            return

        self._barrier = make_obstacle_heatmap(self._obstacles, self._size)
        self._barrier = self._barrier.unsqueeze(0).unsqueeze(0).cuda()

        self._dirty_barrier = False

    def get_barrier(self):
        if self._dirty_barrier:
            self._update_barrier()
        assert(not self._dirty_barrier)
        return self._barrier

    def get_field(self):
        return self._field

    def step(self):
        self._update_barrier()

        self._field = self._wave_kernel(pad_field(self._field, self._padding_parameters))

        self._field *= self._barrier

        assert(self._field.shape == (1, 2, self._size, self._size))

def make_random_field(size, max_num_obstacles=10):
    f = Field(size)
    f.add_obstacles(make_random_obstacles(max_num_obstacles))
    return f