from featurize_audio import crop_audio_from_location_batch
from device_dict import DeviceDict
import torch
import torch.nn as nn

from config import InputConfig, OutputConfig
from reshape_layer import Reshape
from permute_layer import Permute
from log_layer import Log

class EchoLearnNN(nn.Module):
    def __init__(self, input_config, output_config):
        super(EchoLearnNN, self).__init__()

        assert isinstance(input_config, InputConfig)
        assert isinstance(output_config, OutputConfig)
        self._input_config = input_config
        self._output_config = output_config
        
        def makeConvDown(in_channels, out_channels, dims, kernel_size=3, stride=2):
            assert dims in [1, 2]
            ConvType = nn.Conv1d if (dims == 1) else nn.Conv2d
            return nn.Sequential(
                ConvType(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
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
            assert dims in [1, 2, 3]
            ConvType = [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d][dims - 1]
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
            assert dims in [1, 2, 3]
            ConvType = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims - 1]
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

        if self._input_config.using_echo4ch:
            assert self._input_config.format == "spectrogram"
            self.convIn = nn.Sequential(
                makeConvDown(channels_in, 16, dims_in),
                makeConvSame(16, 16, 3, dims_in),
                makeConvDown(16, 16, dims_in),
                makeConvSame(16, 8, 3, dims_in),
                makeConvDown(8, 4, dims_in),
                Reshape((4,32,32), (4*32,32))
            )
            intermediate_width=32
            intermediate_channels=4*32
        elif self._input_config.tof_cropping:
            self.convIn = nn.Sequential(
                makeConvDown(channels_in, 16, dims=1, kernel_size=15),
                makeConvDown(16, 32, kernel_size=16, dims=1),
                makeConvDown(32, 64, kernel_size=16, dims=1),
                Reshape((64,21), (64,21))
            )
            intermediate_width = 21
            intermediate_channels = 64

            # intermediate_width = self._input_config.tof_crop_size
            # intermediate_channels = self._input_config.num_channels
        elif self._input_config.format == "spectrogram":
            assert not self._input_config.using_echo4ch
            self.convIn = nn.Sequential(
                makeConvDown(channels_in, 16, dims_in),
                makeConvDown(16, 32, dims_in),
                makeConvDown(32, 32, dims_in),
                Reshape((32,5,32), (32*5,32))
            )
            intermediate_width=32
            intermediate_channels=32*5
        elif self._input_config.format in ["audioraw", "audiowaveshaped", "gccphat"]:
            assert not self._input_config.using_echo4ch
            self.convIn = nn.Sequential(
                makeConvDown(channels_in, 16, dims_in),
                makeConvDown(16, 16, dims_in, kernel_size=128, stride=2),
                makeConvDown(16, 32, dims_in, kernel_size=128, stride=1),
                makeConvDown(32, 64, dims_in, kernel_size=64, stride=1),
                makeConvDown(64, 64, dims_in, kernel_size=32),
                makeConvDown(64, 128, dims_in, kernel_size=32),
                Reshape((128,45), (128,45)) # safety check
            )
            intermediate_width=45
            intermediate_channels=128
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
        fc_hidden = 128
        fc_outputs = channels_out if self._output_config.tof_cropping else 512

        self.fullyConnected = nn.Sequential(
            makeFullyConnected(fc_inputs + self._output_config.num_implicit_params, fc_hidden),
            makeFullyConnected(fc_hidden, fc_outputs, activation=False)
        )

        if self._output_config.implicit:
            self.final = makeFullyConnected(512, channels_out, activation=False)
        elif self._output_config.tof_cropping:
            # Hahahahaha
            self.final = None
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
        elif dims_out == 3:
            assert self._output_config.resolution == 64
            self.final = nn.Sequential(
                Reshape((512,), (8, 4, 4, 4)),
                makeConvUp(8, 8, 4, dims=3), # 16^3
                makeConvSame(8, 8, 3, dims=3),
                makeConvUp(8, 8, 4, dims=3), # 64^3
                makeConvSame(8, 8, 3, dims=3),
                makeConvSame(8, 8, 3, dims=3),
                makeConvSame(8, channels_out, kernel_size=1, dims=3, activation=False),
                Reshape((channels_out, 64, 64, 64), (channels_out, 64, 64, 64)) # Safety check
            )
        else:
            raise Exception("Unrecognized output format")

        
    def forward(self, d):
        w0  = d['input']
        
        B = w0.shape[0]

        if self._input_config.tof_cropping:
            locations_yx = d['params']
            assert locations_yx.shape == (B, 2)
            w0 = crop_audio_from_location_batch(w0, self._input_config, locations_yx)
            
            # w0_flat = w0.reshape(B, self._input_config.num_channels * self._input_config.tof_crop_size)
            # output = self.fullyConnected(w0_flat)

            conved = self.convIn(w0)
            conved_flat = conved.reshape(B, -1)
            
            output = self.fullyConnected(conved_flat)

            assert output.shape == (B, self._output_config.num_channels)
            output = self.scale_variance(output)
            return DeviceDict({'output': output})

        wx = self.convIn(w0)

        assert len(wx.shape) == 3

        F = wx.shape[1]
        N = wx.shape[2]

        if self._input_config.summary_statistics:
            ls = torch.linspace(0.0, 1.0, N).unsqueeze(0).unsqueeze(0).to(wx.device)
            wx += 0.01

            mom1 = torch.sum(wx * ls, dim=2) / torch.sum(wx, dim=2)
            assert mom1.shape == (B, F)

            mom2 = torch.sqrt(torch.sum((wx * (ls - mom1.unsqueeze(-1)))**2, dim=2))
            assert mom2.shape == (B, F)

            mean = torch.mean(wx, dim=2)
            assert mean.shape == (B, F)

            std = torch.std(wx, dim=2)
            assert std.shape == (B, F)

            summary_stats = torch.stack((mom1, mom2, mean, std), dim=2)
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

        output = self.scale_variance(output)

        return DeviceDict({'output': output})

    def scale_variance(self, x):
        if not self._output_config.predict_variance:
            assert x.shape[1] == 1
            return x
        assert x.shape[1] == 2
        mean = x[:, 0:1]
        pre_variance = x[:, 1:2]

        # variance = torch.exp(pre_variance)

        variance = torch.min(
            torch.stack((
                torch.exp(pre_variance),
                1.0 + torch.abs(pre_variance)
            ),
            dim=1
        ), dim=1)[0]

        return torch.cat((mean, variance), dim=1)

    def save(self, filename):
        print(f"Saving model to \"{filename}\"")
        torch.save(self.state_dict(), filename)

    def restore(self, filename):
        print("Restoring model from \"{}\"".format(filename))
        self.load_state_dict(torch.load(filename))
        self.eval()
