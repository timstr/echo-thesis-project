import torch
import torch.nn as nn

class Permute(nn.Module):
    def __init__(self, new_dim_order):
        super(Permute, self).__init__()
        assert isinstance(new_dim_order, tuple)
        assert list(sorted(new_dim_order)) == list(range(len(new_dim_order)))
        self._new_dim_order = new_dim_order

    def forward(self, x):
        assert(isinstance(x, torch.Tensor))
        B = x.shape[0]
        if len(x.shape[1:]) != len(self._new_dim_order):
            raise Exception(f"Reshape: expected input with {len(self._new_dim_order)} dimensions, but got {len(x.shape[1:])} dimensions instead")
        return x.permute(*([0] + [d + 1 for d in self._new_dim_order]))
