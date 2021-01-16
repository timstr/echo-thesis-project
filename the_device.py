import torch

the_device = "cuda" if torch.cuda.is_available() else "cpu"
