import torch
import random
import math

from tensor_utils import cutout_circle, cutout_rectangle
from wave_kernel import make_wave_kernel

def overlapping(obstacle1, obstacle2):
    t1 = obstacle1[0]
    t2 = obstacle2[0]
    assert(t1 == CIRCLE or t1 == RECTANGLE)
    assert(t2 == CIRCLE or t2 == RECTANGLE)
    args1 = obstacle1[1:]
    args2 = obstacle2[1:]
    if t1 == CIRCLE and t2 == CIRCLE:
        y1, x1, r1 = args1
        y2, x2, r2 = args2
        d = math.hypot(y1 - y2, x1 - x2)
        return d < r1 + r2
    elif t1 == RECTANGLE and t2 == RECTANGLE:
        y1, x1, h1, w1 = args1
        y2, x2, h2, w2 = args2
        return abs(y1 - y2) < (h1 + h2) / 2 and abs(x1 - x2) < (w1 + w2) / 2
    else:
        ry, rx, rw, rh = args1 if t1 == RECTANGLE else args2
        cy, cx, cr = args1 if t1 == CIRCLE else args2
        return abs(ry - cy) < rh / 2 + cr and abs(rx - cx) < rw / 2 + cr

CIRCLE = "Circle"
RECTANGLE = "Rectangle"

class Field():
    def __init__(self, height, width):
        self._height = height
        self._width = width
        self._obstacles = []
        self._dirty_barrier = True
        self._wave_kernel = make_wave_kernel()
        self._field = torch.zeros(
            (1, 2, height, width),
            requires_grad=False,
            dtype=torch.float32
        ).cuda()

    def get_height(self):
        return self._height

    def get_width(self):
        return self._width

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

        self._barrier = torch.ones(
            (1, 2, self._height, self._width),
            requires_grad=False,
            dtype=torch.float32
        ).cuda()

        for o in self._obstacles:
            t = o[0]
            args = o[1:]
            n_args = len(args)
            if (t == CIRCLE):
                assert(n_args == 3)
                cutout_circle(self._barrier, args[0], args[1], args[2])
            elif (t == RECTANGLE):
                assert(n_args == 4)
                cutout_rectangle(self._barrier, args[0], args[1], args[2], args[3])

        # TODO: border fringe?

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
        self._field[...] = self._wave_kernel(self._field) * self._barrier

def make_random_field(height, width, max_num_obstacles=10):
    f = Field(height, width)
    n = random.randint(1, max_num_obstacles)

    def collision(shape):
        if overlapping(shape, (RECTANGLE, 0.5, 0.5, 0.2, 0.2)):
            return True
        for s in f.get_obstacles():
            if (overlapping(s, shape)):
                return True
        return False

    for _ in range(n):
        for _ in range(100):
            # y = random.uniform(0, 0.75)
            y = random.uniform(0, 1)
            x = random.uniform(0, 1)
            if random.random() < 0.5:
                # rectangle
                h = random.uniform(0.01, 0.2)
                w = random.uniform(0.01, 0.2)
                if (collision((RECTANGLE, y, x, h, w))):
                    continue
                f.add_rectangle(y, x, h, w)
                break
            else:
                # circle
                r = random.uniform(0.01, 0.1)
                if collision((CIRCLE, y, x, r)):
                    continue
                f.add_circle(y, x, r)
                break

    return f