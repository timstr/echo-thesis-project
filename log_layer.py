import torch
import torch.nn as nn

from device_dict import DeviceDict

class Log(nn.Module):
    def __init__(self, description=""):
        super(Log, self).__init__()
        assert isinstance(description, str)
        self._description = description
        
    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        print(f"Log Layer - {self._description}")
        print(f"    Batch size:   {x.shape[0]}")
        print(f"    Tensor shape: {x.shape[1:]}")
        print("")
        return x
