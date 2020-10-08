import torch
import torch.nn as nn
import math
from functools import reduce

def prod(iterable, start=1):
    return reduce(lambda a, b: a * b, iterable, start)

from device_dict import DeviceDict

class Reshape(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Reshape, self).__init__()
        assert(isinstance(input_shape, tuple) or isinstance(input_shape, torch.Size))
        assert(isinstance(output_shape, tuple) or isinstance(output_shape, torch.Size))
        assert(prod(input_shape) == prod(output_shape))
        self._input_shape = input_shape
        self._output_shape = output_shape

    def forward(self, x):
        assert(isinstance(x, torch.Tensor))
        B = x.shape[0]
        assert(x.shape[1:] == self._input_shape)
        return x.view(B, *self._output_shape)

class EchoLearnNN(nn.Module):
    def __init__(self, num_input_channels, num_implicit_params, input_format="1D", output_format="scalar", output_resolution=None, predict_variance=False):
        super().__init__()

        self._num_implicit_params = num_implicit_params if num_implicit_params is not None else 0
        self._output_format = output_format
        self._output_resolution = output_resolution
        self._predict_variance = predict_variance
        self._output_dim = 2 if self._predict_variance else 1

        assert(input_format in ["1D", "2D"])

        assert(output_format in ["scalar", "1D", "2D"])

        assert((output_format == "scalar") != isinstance(output_resolution, int))
        
        def makeConvDown(in_channels, out_channels, fmt):
            ConvType = nn.Conv1d if (fmt == "1D") else nn.Conv2d
            return nn.Sequential(
                ConvType(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.ReLU()
            )
        
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

        def makeConvUp(in_channels, out_channels, scale_factor, fmt):
            ConvType = nn.ConvTranspose1d if (fmt == "1D") else nn.ConvTranspose2d
            return ConvType(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=scale_factor,
                stride=scale_factor,
                padding=0,
                output_padding=0
            )

        def makeConvSame(in_channels, out_channels, size, fmt, activation=True):
            assert(size % 2 == 1)
            ConvType = nn.Conv1d if (fmt == "1D") else nn.Conv2d
            conv = ConvType(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=size,
                stride=1,
                padding=((size - 1) // 2)
            )
            if activation:
                return nn.Sequential(
                    conv,
                    nn.ReLU()
                )
            else:
                return conv

        if input_format == "2D":
            self.convDown = nn.Sequential(
                makeConvDown(num_input_channels, 16, input_format),
                makeConvDown(16, 32, input_format),
                makeConvDown(32, 32, input_format),
                makeConvDown(32, 32, input_format),
                Reshape((32, 1, 4), (32, 4))
            )
            fcInputs = 128
        else:
            self.convDown = nn.Sequential(
                makeConvDown(num_input_channels, 16, input_format),
                makeConvDown(16, 16, input_format),
                makeConvDown(16, 32, input_format),
                makeConvDown(32, 64, input_format),
                makeConvDown(64, 64, input_format),
                makeConvDown(128, 128, input_format)
            )

        # summary statistics per channel:
        # - first moment (across pixels)
        # - second moment (across pixels)?
        # - average (across channel values)
        # - variance (accross channel values)
        # Total: 4

        # pooling layer: summary statistics of 32 channels across 64 pixels
        # resulting in 32 channels x 4 statistics = 128 features


        self.fullyConnected = nn.Sequential(
            makeFullyConnected(fcInputs + self._num_implicit_params, 128),
            makeFullyConnected(128, 128),
            makeFullyConnected(128, 256),
            makeFullyConnected(256, 512)
        )

        if self._output_format == "scalar":
            self.final = makeFullyConnected(512, self._output_dim, activation=False)
        elif self._output_format == "1D":
            assert(self._output_resolution >= 16)
            convsFlex = ()
            output_size = 16
            while output_size < self._output_resolution:
                convsFlex += (
                    makeConvUp(32, 32, 2, "1D"),
                    makeConvSame(32, 32, 3, "1D")
                )
                output_size *= 2
            self.final = nn.Sequential(
                Reshape((512,), (64, 8)),    # 8
                makeConvUp(64, 32, 2, "1D"), # 16
                *convsFlex,                      # (final size)
                makeConvSame(32, self._output_dim, 1, "1D", activation=False),
            )
        elif self._output_format == "2D":
            assert(self._output_resolution >= 8)
            convsFlex = ()
            output_size = 16
            while output_size < self._output_resolution:
                convsFlex += (
                    makeConvUp(32, 32, 2, "2D"),
                    makeConvSame(32, 32, 3, "2D")
                )
                output_size *= 2
            self.final = nn.Sequential(
                Reshape((512,), (32, 4, 4)),   # 4x4
                makeConvUp(32, 32, 4, "2D"),   # 16x16
                makeConvSame(32, 32, 3, "2D"), # 16x16
                *convsFlex,                        # (final size)
                makeConvSame(32, self._output_dim, 1, "2D", activation=False),
            )
        else:
            raise Exception("Unrecognized output format")

        
    def forward(self, d):
        w0  = d['input']

        B = w0.shape[0]

        wx = self.convDown(w0)

        assert(len(wx.shape) == 3) # (B, 32, 64))

        F = wx.shape[1]
        N = wx.shape[2]

        ls = torch.linspace(0.0, 1.0, N).unsqueeze(0).unsqueeze(0).to(wx.device)

        mom1 = torch.sum(
            wx * ls,
            dim=2
        )
        assert(mom1.shape == (B, F))

        mom2 = torch.sum(
            (wx * (ls - mom1.unsqueeze(-1)))**2,
            dim=2
        )
        assert(mom2.shape == (B, F))

        mean = torch.mean(
            wx,
            dim=2
        )
        assert(mean.shape == (B, F))

        variance = torch.var(
            wx,
            dim=2
        )
        assert(variance.shape == (B, F))

        summary_stats = torch.stack((
            mom1,
            mom2,
            mean,
            variance
        ), dim=2)
        assert(summary_stats.shape == (B, F, 4))
        summary_stats = summary_stats.reshape(B, F * 4)

        if self._num_implicit_params > 0:
            assert(self._output_format == "scalar")

            implicit_params = d['params']
            
            assert(len(implicit_params.shape) == 3)

            param_batch_size = implicit_params.shape[1]

            assert(implicit_params.shape == (B, param_batch_size, self._num_implicit_params))

            summary_stats_flat = summary_stats.repeat_interleave(param_batch_size, dim=0)
            assert(summary_stats_flat.shape == (B * param_batch_size, F * 4))

            implicit_params_flat = implicit_params.reshape(B * param_batch_size, self._num_implicit_params)

            inputs_flat = torch.cat(
                (summary_stats_flat, implicit_params_flat),
                dim=1
            )

            outputs_flat = self.final(self.fullyConnected(inputs_flat))
            output = outputs_flat.reshape(B, param_batch_size, self._output_dim)
            output = output.permute(0, 2, 1)
            assert(output.shape == (B, self._output_dim, param_batch_size))
        else:
            v0 = summary_stats
            output = self.final(self.fullyConnected(v0))
            assert(output.shape == (B, self._output_dim, self._output_resolution, self._output_resolution))

        assert(output.shape[1] == self._output_dim)
        if (self._predict_variance):
            output = torch.cat((
                output[:, 0:1],
                torch.exp(output[:, 1:2]),
            ), dim=1)

        return DeviceDict({'output': output})
