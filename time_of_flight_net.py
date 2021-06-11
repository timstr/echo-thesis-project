import torch
import torch.nn as nn


class TimeOfFlightNet(nn.Module):
    def __init__(self):
        super(TimeOfFlightNet, self).__init__()

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        return x
