import pickle
import torch
import glob

from featurize import make_obstacle_heatmap
from device_dict import DeviceDict
from featurize import sclog

class WaveSimDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder):
        super(WaveSimDataset).__init__()
        self.data = []
        print("Loading data into memory...")
        for path in glob.glob("{}/example *.pkl".format(data_folder)):
            with open(path, "rb") as file:
                example = pickle.load(file)
            self.data.append(example)
        print("Done.")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        obs, depthmap, echo = self.data[idx]
        echo = torch.tensor(echo).permute(1, 0).float().detach()
        echo_raw = echo
        echo_waveshaped = sclog(torch.tensor(echo))
        depthmap = torch.tensor(depthmap).float().detach()
        assert(echo.shape == (4, 8192))
        assert(depthmap.shape == (128,))
        # mindist = torch.min(depthmap).float().detach()
        # heatmap_first_only = make_first_echo_bump(mindist, 8192).float().detach()
        heatmap = make_obstacle_heatmap(obs, 512, 8192)
        return DeviceDict({
            'obstacles': obs,
            'echo_raw': echo_raw,
            'echo_waveshaped': echo_waveshaped,
            'depthmap': depthmap,
        #   'mindist': mindist,
            'heatmap': heatmap,
        })
