from featurize import implicit_samples_from_dense_output
from config import (
    EmitterConfig,
    InputConfig,
    OutputConfig,
    ReceiverConfig,
    TrainingConfig,
)
import compress_pickle
import torch
import glob
import os

from device_dict import DeviceDict


class Echo4ChDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_config,
        input_config,
        output_config,
        emitter_config,
        receiver_config,
    ):
        """
        output_representation : the representation of expected outputs, must be one of:
                                * "sdf" - signed distance field
                                * "heatmap" - binary heatmap
                                * "depthmap" - line-of-sight distance, e.g. radarplot
        """
        super(Echo4ChDataset).__init__()

        assert isinstance(training_config, TrainingConfig)
        assert isinstance(input_config, InputConfig)
        assert isinstance(output_config, OutputConfig)
        assert isinstance(emitter_config, EmitterConfig)
        assert isinstance(receiver_config, ReceiverConfig)
        self._training_config = training_config
        self._input_config = input_config
        self._output_config = output_config
        self._emitter_config = emitter_config
        self._receiver_config = receiver_config

        assert self._emitter_config.arrangement == "mono"
        assert self._emitter_config.format == "sweep"

        # NOTE: technically the long- and short-window spectrograms are from the same 4 receivers, but this doesn't make a difference
        assert self._receiver_config.count == 8
        assert self._receiver_config.arrangement == "grid"

        assert self._input_config.format == "spectrogram"
        assert self._input_config.num_channels == 8

        assert self._output_config.format in ["depthmap", "heatmap"]
        assert self._output_config.resolution == 64

        self._rootpath = os.environ.get("ECHO4CH_DATASET")

        if self._rootpath is None or not os.path.exists(self._rootpath):
            raise Exception(
                "Please set the ECHO4CH_DATASET environment variable to point to the WaveSim dataset root"
            )

        print(
            f'NOTE: dataset files will be loaded continuously from "{self._rootpath}", because echo4ch is just too big to fit in memory'
        )

        self._filenames = sorted(glob.glob("{}/example *.pkl".format(self._rootpath)))
        num_files = len(self._filenames)
        if num_files == 0:
            raise Exception(
                "The ECHO4CH_DATASET environment variable points to a folder which contains no example files. Windows users: did you accidentally include quotes in the environment variable?"
            )
        max_examples = self._training_config.max_examples
        if max_examples is not None:
            if num_files >= max_examples:
                num_files = num_files[:max_examples]
            else:
                print("Warning! Fewer matching examples were found than were requested")
                print("    Expected: ", max_examples)
                print("    Actual:   ", num_files)

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, idx):
        path = self._filenames[idx]
        with open(path, "rb") as file:
            data = compress_pickle.load(file, compression="gzip")
        spectrograms = torch.tensor(data["spectrograms"], dtype=torch.float) / 255.0
        occupancy = torch.tensor(data["occupancy"], dtype=torch.float) / 255.0
        depthmap = torch.tensor(data["depthmap"], dtype=torch.float) / 255.0

        dense_output = (
            occupancy if (self._output_config.format == "heatmap") else depthmap
        )

        theDict = {
            "input": spectrograms,
            "gt_heatmap": occupancy,
            "gt_depthmap": depthmap,
        }

        if self._output_config.implicit:
            spe = self._training_config.samples_per_example
            params, values = implicit_samples_from_dense_output(dense_output, spe)
            theDict["params"] = params
            theDict["output"] = values
        else:
            theDict["output"] = dense_output

        return DeviceDict(theDict)
