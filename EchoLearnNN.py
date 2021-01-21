from device_dict import DeviceDict
import torch
import torch.nn as nn

from dataset_config import InputConfig, OutputConfig
from reshape_layer import Reshape
from permute_layer import Permute
from log_layer import Log

class EchoLearnNN(nn.Module):
    def __init__(self, input_config, output_config):
        super().__init__()

        assert isinstance(input_config, InputConfig)
        assert isinstance(output_config, OutputConfig)
        self._input_config = input_config
        self._output_config = output_config
        
        def makeConvDown(in_channels, out_channels, dims):
            assert dims in [1, 2]
            ConvType = nn.Conv1d if (dims == 1) else nn.Conv2d
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

        def makeConvUp(in_channels, out_channels, scale_factor, dims):
            assert dims in [1, 2]
            ConvType = nn.ConvTranspose1d if (dims == 1) else nn.ConvTranspose2d
            return ConvType(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=scale_factor,
                stride=scale_factor,
                padding=0,
                output_padding=0
            )

        def makeConvSame(in_channels, out_channels, kernel_size, dims, activation=True):
            assert kernel_size % 2 == 1
            assert dims in [1, 2]
            ConvType = nn.Conv1d if (dims == 1) else nn.Conv2d
            conv = ConvType(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2)
            )
            if activation:
                return nn.Sequential(
                    conv,
                    nn.ReLU()
                )
            else:
                return conv

        dims_in = self._input_config.dims
        dims_out = self._output_config.dims
        channels_in = self._input_config.num_channels
        channels_out = self._output_config.num_channels


        if self._input_config.format == "spectrogram":
            self.convIn = nn.Sequential(
                makeConvDown(channels_in, 16, dims_in),
                makeConvDown(16, 32, dims_in),
                makeConvDown(32, 32, dims_in),
                makeConvDown(32, 64, dims_in),
                makeConvDown(64, 64, dims_in),
                Reshape((64,2,8), (128,8))
            )
            intermediate_width=8
            intermediate_channels=128
        elif self._input_config.format in ["audioraw", "audiowaveshaped"]:
            self.convIn = nn.Sequential(
                makeConvDown(channels_in, 16, dims_in),
                makeConvDown(16, 16, dims_in),
                makeConvDown(16, 32, dims_in),
                makeConvDown(32, 32, dims_in),
                makeConvDown(32, 32, dims_in),
                makeConvDown(32, 32, dims_in),
                Reshape((32,32), (32,32)) # safety check
            )
            intermediate_width=32
            intermediate_channels=32
        else:
            raise Exception(f"Unrecognized input format: '{self._input_config.format}'")

        # summary statistics per channel:
        # - first moment (across pixels)
        # - second moment (across pixels)
        # - average (across channel values)
        # - variance (accross channel values)
        # Total: 4
        num_summary_stats = 4
        
        fc_inputs = (intermediate_channels * num_summary_stats) if self._input_config.summary_statistics else (intermediate_channels * intermediate_width)

        self.fullyConnected = nn.Sequential(
            makeFullyConnected(fc_inputs + self._output_config.num_implicit_params, 128),
            makeFullyConnected(128, 512)
        )

        if self._output_config.implicit:
            self.final = makeFullyConnected(512, channels_out, activation=False)
        elif dims_out == 1:
            assert self._output_config.resolution >= 16
            convsFlex = ()
            output_size = 16
            while output_size < self._output_config.resolution:
                convsFlex += (
                    makeConvUp(32, 32, 2, dims=1),
                    makeConvSame(32, 32, 3, dims=1)
                )
                output_size *= 2
            self.final = nn.Sequential(
                Reshape((512,), (64, 8)),    # 8
                makeConvUp(64, 32, 2, dims=1), # 16
                *convsFlex,                      # (final size)
                makeConvSame(32, channels_out, kernel_size=1, dims=1, activation=False),
            )
        elif dims_out == 2:
            assert self._output_config.resolution >= 8
            convsFlex = ()
            output_size = 16
            while output_size < self._output_config.resolution:
                convsFlex += (
                    makeConvUp(32, 32, 2, dims=2),
                    makeConvSame(32, 32, 3, dims=2)
                )
                output_size *= 2
            self.final = nn.Sequential(
                Reshape((512,), (32, 4, 4)),   # 4x4
                makeConvUp(32, 32, 4, dims=2),   # 16x16
                makeConvSame(32, 32, 3, dims=2), # 16x16
                *convsFlex,                        # (final size)
                makeConvSame(32, channels_out, kernel_size=1, dims=2, activation=False),
            )
        else:
            raise Exception("Unrecognized output format")

        
    def forward(self, d):
        w0  = d['input']
        
        B = w0.shape[0]

        wx = self.convIn(w0)

        assert len(wx.shape) == 3

        F = wx.shape[1]
        N = wx.shape[2]

        if self._input_config.summary_statistics:
            ls = torch.linspace(0.0, 1.0, N).unsqueeze(0).unsqueeze(0).to(wx.device)
            wx += 0.01

            mom1 = torch.sum(wx * ls, dim=2) / torch.sum(wx, dim=2)
            assert mom1.shape == (B, F)

            mom2 = torch.sum((wx * (ls - mom1.unsqueeze(-1)))**2, dim=2)
            assert mom2.shape == (B, F)

            mean = torch.mean(wx, dim=2)
            assert mean.shape == (B, F)

            variance = torch.var(wx, dim=2)
            assert variance.shape == (B, F)

            summary_stats = torch.stack((mom1, mom2, mean, variance), dim=2)
            assert summary_stats.shape == (B, F, 4)

            fc_input = summary_stats.reshape(B, F * 4)
        else:
            fc_input = wx.reshape(B, -1)

        res = self._output_config.resolution
        out_dims = self._output_config.dims
        out_channels = self._output_config.num_channels
        num_params = self._output_config.num_implicit_params

        if self._output_config.implicit:
            implicit_params = d['params']
            
            assert len(implicit_params.shape) == 3

            param_batch_size = implicit_params.shape[1]

            assert implicit_params.shape == (B, param_batch_size, num_params)

            summary_stats_flat = fc_input.repeat_interleave(param_batch_size, dim=0)

            implicit_params_flat = implicit_params.reshape(B * param_batch_size, num_params)

            inputs_flat = torch.cat(
                (summary_stats_flat, implicit_params_flat),
                dim=1
            )

            v0 = self.fullyConnected(inputs_flat)
            outputs_flat = self.final(v0)
            output = outputs_flat.reshape(B, param_batch_size, out_channels)
            output = output.permute(0, 2, 1)
            assert output.shape == (B, out_channels, param_batch_size)
        else:
            v0 = fc_input
            v1 = self.fullyConnected(v0)
            output = self.final(v1)
            assert output.shape == ((B, out_channels) + ((res,) * out_dims))

        if (self._output_config.predict_variance):
            mean = output[:, 0:1]
            pre_variance = output[:, 1:2]

            variance = torch.exp(pre_variance)

            # variance = torch.min(
            #     torch.stack((
            #         torch.exp(pre_variance),
            #         1.0 + torch.abs(pre_variance)
            #     ),
            #     dim=1
            # ), dim=1)[0]

            output = torch.cat((mean, variance), dim=1)

        return DeviceDict({'output': output})

    def save(self, filename):
        print(f"Saving model to \"{filename}\"")
        torch.save(self.state_dict(), filename)

    def restore(self, filename):
        print("Restoring model from \"{}\"".format(filename))
        self.load_state_dict(torch.load(filename))
        self.eval()
