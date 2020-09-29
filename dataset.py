import pickle
import torch
import glob

from featurize import sclog, make_sdf_image_gt, make_depthmap_gt, make_heatmap_image_gt, shortest_distance_to_obstacles, line_of_sight_from_bottom_up, CIRCLE, RECTANGLE
from device_dict import DeviceDict
from featurize import sclog
from progress_bar import progress_bar
from featurize_audio import make_spectrogram

# dataset formats
inputformat_ar = "audioraw"
inputformat_aw = "audiowaveshaped"
inputformat_sg = "spectrogram"
inputformat_all = [inputformat_ar, inputformat_aw, inputformat_sg]

# dataset formats
outputformat_sdf = "sdf"
outputformat_hm = "heatmap"
outputformat_dm = "depthmap"
outputformat_all = [outputformat_sdf, outputformat_hm, outputformat_dm]

# num parameters
outputformat_params_map = {
    outputformat_sdf: 2,
    outputformat_hm: 2,
    outputformat_dm: 1
}

class WaveSimDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, samples_per_example=1024, num_examples=None, max_obstacles=None, receiver_indices=range(64), circles_only=False, input_representation="audiowaveshaped",
        output_representation="sdf", implicit_function=True, dense_output_resolution=32):
        # TODO: document this further
        """
            output_representation : the representation of expected outputs, must be one of:
                                    * "sdf" - signed distance field
                                    * "heatmap" - binary heatmap
                                    * "depthmap" - line-of-sight distance, e.g. radarplot
        """
        super(WaveSimDataset).__init__()

        def allCircles(obs):
            for o in obs:
                if o[0] != CIRCLE:
                    return False
            return True

        self._spe = samples_per_example
        self._data = []
        self._dense_output_cache = {}

        assert(len(receiver_indices) <= 64)
        assert(set(receiver_indices).issubset(range(64)))
        self._receiver_indices = receiver_indices

        assert(input_representation in inputformat_all)
        assert(output_representation in outputformat_all)
        self._input_representation = input_representation
        self._output_representation = output_representation
        self._implicit_function = implicit_function
        self._dense_output_resolution = dense_output_resolution

        print("Loading data into memory from \"{}\"...".format(data_folder))
        
        filenames = sorted(glob.glob("{}/example *.pkl".format(data_folder)))
        num_files = len(filenames)
        if num_examples is not None:
            num_files = min(num_files, num_examples)
        for i, path in enumerate(filenames):
            with open(path, "rb") as file:
                obs, echo = pickle.load(file)
                echo = torch.tensor(echo).permute(1, 0).float()
                echo = echo[receiver_indices, :]
            if max_obstacles is not None and len(obs) > max_obstacles:
                continue
            if circles_only and not allCircles(obs):
                continue
            progress_bar(len(self._data), num_files)
            self._data.append((obs, echo))
            if num_examples is not None and len(self._data) == num_examples:
                break
        print(" Done.")
        if num_examples is not None and len(self._data) != num_examples:
            print("Warning! Fewer matching examples were found than were requested")
            print("    Expected: ", num_examples)
            print("    Actual:   ", len(self._data))
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        obs, echo = self._data[idx]
        
        echo_raw = torch.tensor(echo)
        if self._input_representation == inputformat_ar:
            the_input = echo_raw
        elif self._input_representation == inputformat_aw:
            the_input = sclog(echo_raw)
        elif self._input_representation == inputformat_sg:
            the_input = make_spectrogram(echo_raw)
        
        theDict = {
            'obstacles_list': obs,
            'input': the_input
        }

        if self._implicit_function:
            # input parameters, scalar output
            n_params = outputformat_params_map[self._output_representation]
            params = torch.rand(self._spe, n_params, requires_grad=True)
            output = torch.zeros(self._spe)
            for i in range(self._spe):
                p = params[i]
                if self._output_representation == outputformat_sdf:
                    v = shortest_distance_to_obstacles(obs, p[0], p[1])
                elif self._output_representation == outputformat_hm:
                    v = 1.0 if shortest_distance_to_obstacles(obs, p[0], p[1]) < 1e-6 else 0.0
                elif self._output_representation == outputformat_dm:
                    v = line_of_sight_from_bottom_up(obs, p[0])
                output[i] = torch.tensor(v, requires_grad=True)

            theDict["params"] = params
            theDict["output"] = output
        else: 
            # dense output
            if idx in self._dense_output_cache:
                output = self._dense_output_cache[idx]
            else:
                res = self._dense_output_resolution
                if self._output_representation == outputformat_sdf:
                    output = make_sdf_image_gt(theDict, res)
                    assert(output.shape == (res, res))
                elif self._output_representation == outputformat_hm:
                    output = make_heatmap_image_gt(theDict, res)
                    assert(output.shape == (res, res))
                elif self._output_representation == outputformat_dm:
                    output = make_depthmap_gt(theDict, res)
                    assert(len(output.shape) == 1)
                    assert(output.shape[0] == res)
                self._dense_output_cache[idx] = output
            theDict["output"] = output
        
        return DeviceDict(theDict)
