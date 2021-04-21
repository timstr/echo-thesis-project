import fix_dead_command_line

import torch
import math
from os import abort


def sanitycheck(x, desc):
    mag = torch.max(torch.abs(x)).item()
    if not math.isfinite(mag):
        print(f"OH NO! Non-finite magnitude {mag} detected during {desc}")
        print("Time to die")
        abort()
    if mag > 50.0:
        print(f"WATCH OUT! Large magnitude {mag} detected during {desc}")


class SanityCheckLayer(torch.nn.Module):
    def __init__(self, description="", module=None):
        super(SanityCheckLayer, self).__init__()
        assert isinstance(description, str)
        self._description = description
        assert module is None or isinstance(module, torch.nn.Module)
        self._module = module

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        sanitycheck(x, self._description)
        if self._module is not None:
            for i, p in enumerate(self._module.parameters()):
                sanitycheck(p.data.detach(), f"{self._description}, parameter {i}")
            x = self._module(x)
            sanitycheck(x, f"{self._description} output")
            return x
        return x
