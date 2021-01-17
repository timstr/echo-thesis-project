import torch
import torch.nn as nn
from functools import reduce

def prod(iterable, start=1):
    return reduce(lambda a, b: a * b, iterable, start)

class Reshape(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Reshape, self).__init__()
        assert(isinstance(input_shape, tuple) or isinstance(input_shape, torch.Size))
        assert(isinstance(output_shape, tuple) or isinstance(output_shape, torch.Size))
        assert(prod(input_shape) == prod(output_shape))
        self._input_shape = input_shape
        self._output_shape = output_shape

    def forward(self, x):
        assert(isinstance(x, torch.Tensor))
        B = x.shape[0]
        if x.shape[1:] != self._input_shape:
            raise Exception(f"Reshape: expected input size {self._input_shape} but got {tuple(x.shape[1:])} instead")
        return x.view(B, *self._output_shape)
