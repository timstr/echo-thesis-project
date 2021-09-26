import os
import torch

env_compute_device = os.environ.get("COMPUTE_DEVICE")

if env_compute_device is not None:
    s_compute_device = env_compute_device
else:
    s_compute_device = "cuda" if torch.cuda.is_available() else "cpu"


def get_compute_device():
    return s_compute_device
