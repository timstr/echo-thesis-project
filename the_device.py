import torch

the_device = "cuda" if torch.cuda.is_available() else "cpu"

what_my_gpu_can_handle = 128**2
