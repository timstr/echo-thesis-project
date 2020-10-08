import pickle
import torch
import glob

from featurize import sclog, make_sdf_image_gt, make_depthmap_gt, make_heatmap_image_gt, make_implicit_params_train, make_implicit_outputs, make_dense_outputs, CIRCLE, RECTANGLE
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

class WaveSimDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, samples_per_example, num_examples, max_obstacles, receiver_indices, circles_only, input_representation,
        output_representation, implicit_function, dense_output_resolution):
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
        obs, echo_raw = self._data[idx]
        
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
            params = make_implicit_params_train(self._spe, self._output_representation)
            output = make_implicit_outputs(obs, params, self._output_representation)

            theDict["params"] = params
            theDict["output"] = output
        else: 
            # dense output
            if idx in self._dense_output_cache:
                output = self._dense_output_cache[idx]
            else:
                output = make_dense_outputs(obs, self._output_representation, self._dense_output_resolution)
                self._dense_output_cache[idx] = output
            theDict["output"] = output
        
        return DeviceDict(theDict)
