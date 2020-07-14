import pickle
import torch
import glob

from featurize import make_obstacle_heatmap, permute_example, shortest_distance_to_obstacles
from device_dict import DeviceDict
from featurize import sclog #, center_and_undelay_signal, add_signal_heatmap
from field import make_obstacle_map, CIRCLE, RECTANGLE



class WaveSimDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, permute=True, samples_per_example=1024):
        super(WaveSimDataset).__init__()
        self._permute = permute
        self._spe = samples_per_example
        self._data = []
        print("Loading data into memory...")
        for path in sorted(glob.glob("{}/example *.pkl".format(data_folder))):
            with open(path, "rb") as file:
                # obs, depthmap, echo = pickle.load(file)
                obs, echo = pickle.load(file)
                echo = torch.tensor(echo).permute(1, 0).float().detach()
            self._data.append((obs, echo))
        print("Done.")
    
    def __len__(self):
        return len(self._data) * (8 if self._permute else 1) * self._spe
    
    def __getitem__(self, idx):
        idx = idx // self._spe
        if (self._permute):
            permutation = idx % 8
            idx = idx // 8
        obs, echo = self._data[idx]

        if (self._permute):
            obs, echo = permute_example(obs, echo, permutation)
        
        sdf_yx = torch.rand(2)

        sdf_value = torch.tensor([
            shortest_distance_to_obstacles(obs, sdf_yx[0], sdf_yx[1])
        ])

        echo_raw = echo
        echo_waveshaped = sclog(torch.tensor(echo))
        obstacles_map = make_obstacle_map(obs, 32, 32)
        # assert(echo.shape == (4, 4096))
        # assert(echo.shape == (64, 8192))
        echo_len = echo.shape[1]
        heatmap = make_obstacle_heatmap(obs, 512, echo_len)
        return DeviceDict({
            'obstacles_list': obs,
            'obstacles': obstacles_map,
            'echo_raw': echo_raw,
            'echo_waveshaped': echo_waveshaped, # add_signal_heatmap(echo_waveshaped, sdf_yx[0], sdf_yx[1]),
            'heatmap': heatmap,
            'sdf_location': sdf_yx,
            'sdf_value': sdf_value
        })
