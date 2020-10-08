import torch
import torch.nn as nn
import math

from device_dict import DeviceDict

class SimpleNN(nn.Module):
    def __init__(self, num_input_channels, num_implicit_params, input_format="1D", output_format="scalar", output_resolution=None, predict_variance=False):
        super().__init__()

        self._num_implicit_params = num_implicit_params if num_implicit_params is not None else 0
        self._output_format = output_format
        self._output_resolution = output_resolution if output_resolution is not None else 1
        self._predict_variance = predict_variance

        assert(False) # TODO: implement predicting variance

        self._param_repetitions = 1

        assert(input_format in ["1D", "2D"])

        assert(output_format in ["scalar", "1D", "2D"])

        assert((output_format == "scalar") != isinstance(output_resolution, int))

        # assert((output_format == "scalar") == (self._output_resolution == 1))
        # assert((output_format == "scalar") == (self._num_implicit_params != 0))

        assert((output_resolution is None) != isinstance(output_resolution, int))

        def makeFullyConnected(in_features, out_features, activation=True):
            lin = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=True
            )
            if activation:
                return nn.Sequential(
                    lin,
                    nn.ReLU()
                )
            else:
                return lin

        if self._output_format == "scalar":
            self._num_outputs = 1
        elif self._output_format == "1D":
            self._num_outputs = output_resolution
        elif self._output_format == "2D":
            self._num_outputs = output_resolution**2

        self._num_inputs = 4096 * num_input_channels

        self.fc1 = makeFullyConnected(self._num_inputs, 128)
        self.fc2 = makeFullyConnected(128, 128)
        self.fc3 = makeFullyConnected(128, 32)
        self.fc4 = makeFullyConnected(32 + self._num_implicit_params, 128)
        self.fc5 = makeFullyConnected(128, 128)
        self.fc6 = makeFullyConnected(128, self._num_outputs, activation=False)
        
    def forward(self, d):
        w  = d['input']

        B = w.shape[0]

        w = w.reshape(B, -1)

        assert(w.shape == (B, self._num_inputs))

        x1 = self.fc1(w)
        x2 = self.fc2(x1)
        x3 = self.fc3(x2)

        if self._num_implicit_params > 0:
            implicit_params = d['params']
            assert(implicit_params.shape == (B, self._num_implicit_params))
            x3 = torch.cat(
                (x3, implicit_params),
                dim=1
            )

        x4 = self.fc4(x3)
        x5 = self.fc5(x4)
        x6 = self.fc6(x5)

        xfinal = x6

        assert(xfinal.shape == (B, self._num_outputs))

        if self._output_format == "scalar":
            output = xfinal
        elif self._output_format == "1D":
            output = xfinal
        elif self._output_format == "2D":
            output = xfinal.reshape(B, self._output_resolution, self._output_resolution)

        return DeviceDict({'output': output})