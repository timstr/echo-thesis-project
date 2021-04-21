import torch

# utility dictionary that can move tensor values between devices via the 'to(device)' function
from collections import OrderedDict


class DeviceDict(dict):
    # following https://stackoverflow.com/questions/3387691/how-to-perfectly-override-a-dict
    def __init__(self, *args):
        super(DeviceDict, self).__init__(*args)

    def to(self, device):
        dd = DeviceDict()  # return shallow copy
        for k, v in self.items():
            if torch.is_tensor(v):
                dd[k] = v.to(device)
            else:
                dd[k] = v
        return dd
