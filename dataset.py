import pickle
import torch
import glob
import os

from convert_input import combine_emitted_signals, transform_received_signals
from config import (
    EmitterConfig,
    InputConfig,
    OutputConfig,
    ReceiverConfig,
    TrainingConfig,
    example_should_be_used,
    output_format_depthmap,
    output_format_sdf,
    output_format_heatmap,
    input_format_audioraw,
    input_format_audiowaveshaped,
    input_format_gcc,
    input_format_gccphat,
)
from featurize import (
    make_implicit_params_train,
    make_implicit_outputs,
    make_dense_outputs,
)
from device_dict import DeviceDict
from progress_bar import progress_bar


class WaveSimDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_config,
        input_config,
        output_config,
        emitter_config,
        receiver_config,
    ):
        super(WaveSimDataset).__init__()

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

        self._data = []
        self._dense_output_cache = {}
        self._rootpath = os.environ.get("WAVESIM_DATASET")

        if self._rootpath is None or not os.path.exists(self._rootpath):
            raise Exception(
                "Please set the WAVESIM_DATASET environment variable to point to the WaveSim dataset root"
            )

        print('Loading data into memory from "{}"...'.format(self._rootpath))

        filenames = sorted(glob.glob("{}/example *.pkl".format(self._rootpath)))
        num_files = len(filenames)
        if num_files == 0:
            raise Exception(
                "The WAVESIM_DATASET environment variable points to a folder which contains no example files. Windows users: did you accidentally include quotes in the environment variable?"
            )
        if self._training_config.max_examples is not None:
            num_files = min(num_files, self._training_config.max_examples)
        for i, path in enumerate(filenames):
            with open(path, "rb") as file:
                data = pickle.load(file)
            impulse_responses = torch.tensor(
                data["impulse_responses"], dtype=torch.float
            )
            obstacles = data["obstacles"]
            occlusion = data["occlusion"]
            if not example_should_be_used(obstacles, occlusion, self._training_config):
                continue
            progress_bar(len(self._data), num_files)
            self._data.append((obstacles, impulse_responses))
            if (
                self._training_config.max_examples is not None
                and len(self._data) >= self._training_config.max_examples
            ):
                break
        print(" Done.")
        if (
            self._training_config.max_examples is not None
            and len(self._data) != self._training_config.max_examples
        ):
            print("Warning! Fewer matching examples were found than were requested")
            print("    Expected: ", self._training_config.max_examples)
            print("    Actual:   ", len(self._data))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        obstacles, impulse_responses = self._data[idx]

        received_signals = combine_emitted_signals(
            impulse_responses, self._emitter_config, self._receiver_config
        )
        the_input = transform_received_signals(received_signals, self._input_config)

        theDict = {"obstacles_list": obstacles, "input": the_input}

        if self._output_config.tof_cropping:
            assert self._input_config.tof_cropping
            assert self._output_config.dims == 2
            assert self._output_config.format in [
                output_format_sdf,
                output_format_heatmap,
            ]
            assert self._input_config.format in [
                input_format_audioraw,
                input_format_audiowaveshaped,
                input_format_gcc,
                input_format_gccphat,
            ]

            sample_location_yx = torch.rand(2)
            theDict["output"] = make_implicit_outputs(
                obstacles, sample_location_yx.unsqueeze(0), self._output_config.format
            ).squeeze(0)
            theDict["params"] = sample_location_yx

        elif self._output_config.implicit:
            spe = self._training_config.samples_per_example
            params = make_implicit_params_train(spe, self._output_config.format)
            output = make_implicit_outputs(
                obstacles, params, self._output_config.format
            )

            if (
                self._training_config.importance_sampling
                and self._output_config.format != output_format_depthmap
            ):
                # TODO: move this to a separate function for importance-sampled implicit location
                def get_filter(x, params):
                    sdf = (
                        x
                        if self._output_config == output_format_sdf
                        else make_implicit_outputs(obstacles, params, output_format_sdf)
                    )
                    return 0.1 + 0.9 * (
                        1.0 - torch.round(torch.clamp(torch.abs(10.0 * sdf), max=1.0))
                    )

                def maybe_accept(x, params):
                    y = get_filter(x, params)
                    assert torch.all(y >= 0.0) and torch.all(y <= 1.0)
                    r = torch.rand_like(y)
                    mask = y >= r
                    return mask

                mask = torch.ones_like(output).bool()
                iterations = 0
                # and torch.any(mask): # NOTE: torch.any is slow (probably because it computes a reduction over the entire tensor)
                while iterations < 8:
                    params[mask] = make_implicit_params_train(
                        spe, self._output_config.format
                    )[mask]
                    output[mask] = make_implicit_outputs(
                        obstacles, params, self._output_config.format
                    )[mask]
                    m = maybe_accept(output, params)
                    mask[m] = False
                    iterations += 1

            theDict["params"] = params
            theDict["output"] = output
        else:
            # dense output
            if idx in self._dense_output_cache:
                output = self._dense_output_cache[idx]
            else:
                output = make_dense_outputs(
                    obstacles,
                    self._output_config.format,
                    self._output_config.resolution,
                )
                self._dense_output_cache[idx] = output
            theDict["output"] = output

        return DeviceDict(theDict)
