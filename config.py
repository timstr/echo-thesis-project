import math
import numpy as np

from wavesim_params import make_emitter_indices, make_receiver_indices, wavesim_duration
from shape_types import CIRCLE

emitter_beep_sweep_len = wavesim_duration // 16
emitter_sequential_delay = wavesim_duration // 8

def make_emitter_signal(frequency_index, delay, format):
    assert frequency_index in range(5)
    assert delay < wavesim_duration
    assert format in ["impulse", "beep", "sweep"]
    # frequency (relative to sampling frequency)
    freq = math.pow(2.0, frequency_index / 5.0) / 32.0 # pentatonic scale ftw
    sig_len = min(max(wavesim_duration - delay, 0), emitter_beep_sweep_len)
    
    out = np.zeros((wavesim_duration,))
    out_sig = out[delay:delay+sig_len]
    if format == "impulse":
        out[delay] = 1.0
    elif format == "beep":
        t = np.linspace(0.0, 2.0 * np.pi * sig_len, sig_len)
        out_sig[...] = 1.0 - 2.0 * np.round(0.5 + 0.5 * np.sin(t * freq))
    elif format == "sweep":
        p = 0.0
        for i, t in enumerate(np.linspace(0.0, 1.0, sig_len)):
            k = math.pow(2.0, 1.0 - 2.0 * t)
            p += freq * k
            out_sig[i] = math.sin(p * 2.0 * np.pi)
    else:
        raise Exception("Unrecognized signal format")
    return out

class EmitterConfig:
    def __init__(self, arrangement="mono", format="sweep", sequential=False, sameFrequency=False):
        assert arrangement in ["mono", "stereo", "surround"]
        assert format in ["impulse", "beep", "sweep"]
        assert isinstance(sequential, bool)
        assert isinstance(sameFrequency, bool)
        self.arrangement = arrangement
        self.format = format
        self.sequential = sequential
        self.sameFrequency = sameFrequency

        self.indices = make_emitter_indices(self.arrangement)

        self.emitted_signals = [make_emitter_signal(
            frequency_index=(0 if self.sameFrequency else i),
            delay=(i * emitter_sequential_delay if self.sequential else 0),
            format=self.format
        ) for i in range(len(self.indices))]
    
    def print(self):
        print(f"Emitter Configuration:")
        print(f"  arrangement              : {self.arrangement}")
        print(f"  format                   : {self.format}")
        print(f"  sequential?              : {self.sequential}")
        print(f"  same frequency?          : {self.sequential}")

class ReceiverConfig:
    def __init__(self, arrangement="grid", count=8):
        assert (arrangement, count) in [
            ("flat", 1),
            ("flat", 2),
            ("flat", 4),
            ("grid", 4),
            ("grid", 8)
        ]
        self.arrangement = arrangement
        self.count = count
        self.indices = make_receiver_indices(self.count, self.arrangement)

    def print(self):
        print(f"Receiver Configuration:")
        print(f"  arrangement              : {self.arrangement}")
        print(f"  count                    : {self.count}")


class InputConfig:
    def __init__(self, emitter_config, receiver_config, format="spectrogram", summary_statistics=True, using_echo4ch=False, tof_crop_size=None):
        assert format in ["audioraw", "audiowaveshaped", "spectrogram", "gcc", "gccphat"]
        assert isinstance(emitter_config, EmitterConfig)
        assert isinstance(receiver_config, ReceiverConfig)
        assert tof_crop_size is None or isinstance(tof_crop_size, int)

        self.format = format
        # HACK to simplify emitter arrangements (only mono is being used right now)
        assert len(emitter_config.emitted_signals) == 1
        self.emitter_config = emitter_config
        self.receiver_config = receiver_config
        self.emitted_signal = emitter_config.emitted_signals[0]
        self.num_channels = receiver_config.count
        self.summary_statistics = summary_statistics
        self.using_echo4ch = using_echo4ch
        self.tof_cropping = tof_crop_size is not None
        self.tof_crop_size = tof_crop_size

        self.dims = 2 if (self.format in ["spectrogram"]) else 1

        if self.using_echo4ch:
            assert self.format == "spectrogram"
            assert self.num_channels == 8
            assert not self.tof_cropping

    def print(self):
        print(f"Input Configuration:")
        print(f"  format                   : {self.format}")
        print(f"  summary statistics?      : {self.summary_statistics}")
        print(f"  -> channels              : {self.num_channels}")
        print(f"  -> dimensions            : {self.dims}")
        print(f"  time-of-flight cropping? : {self.tof_cropping}")
        print(f"  time-of-flight crop size : {self.tof_crop_size}")

class OutputConfig:
    def __init__(self, format="sdf", implicit=True, predict_variance=True, tof_cropping=False, resolution=1024, using_echo4ch=False):
        assert format in ["depthmap", "heatmap", "sdf"]
        assert isinstance(implicit, bool)
        assert isinstance(predict_variance, bool)
        assert isinstance(using_echo4ch, bool)
        assert isinstance(tof_cropping, bool)
        assert not (implicit and tof_cropping)
        assert resolution in [32, 64, 128, 256, 512, 1024]
        self.format = format
        self.implicit = implicit
        self.predict_variance = predict_variance
        self.tof_cropping = tof_cropping
        self.resolution = resolution
        self.using_echo4ch = using_echo4ch

        if using_echo4ch:
            assert format != "sdf"
            assert resolution == 64
            self.dims = 2 if format == "depthmap" else 3
        else:
            self.dims = 1 if format == "depthmap" else 2
        self.num_implicit_params = 0 if not implicit else self.dims
        self.num_channels = 2 if predict_variance else 1
    
    def print(self):
        print(f"Output Configuration:")
        print(f"  format                   : {self.format}")
        print(f"  implicit function?       : {self.implicit}")
        print(f"  predict variance?        : {self.predict_variance}")
        print(f"  time-of-flight cropping? : {self.tof_cropping}")
        print(f"  resolution               : {self.resolution}")
        print(f"  -> dimensions            : {self.dims}")
        print(f"  -> implicit parameters   : {self.num_implicit_params}")
        print(f"  -> channels              : {self.num_channels}")


class TrainingConfig:
    def __init__(self, max_examples=None, max_obstacles=None, circles_only=False, allow_occlusions=True, importance_sampling=False, samples_per_example=128):
        assert max_examples is None or isinstance(max_examples, int)
        assert max_obstacles is None or isinstance(max_obstacles, int)
        assert isinstance(circles_only, bool)
        assert isinstance(allow_occlusions, bool)
        assert isinstance(importance_sampling, bool)
        assert isinstance(importance_sampling, bool)
        assert isinstance(samples_per_example, int)
        
        self.max_examples = max_examples
        self.max_obstacles = max_obstacles
        self.circles_only = circles_only
        self.allow_occlusions = allow_occlusions
        self.importance_sampling = importance_sampling
        self.samples_per_example = samples_per_example

    def print(self):
        def someOrAll(n):
            return "no limit" if n is None else str(n)
        print(f"Training Configuration:")
        print(f"  maximum examples         : {someOrAll(self.max_examples)}")
        print(f"  maximum obstacles        : {someOrAll(self.max_obstacles)}")
        print(f"  circles only?            : {self.circles_only}")
        print(f"  allow occlusions?        : {self.allow_occlusions}")
        print(f"  importance sampling?     : {self.importance_sampling}")
        print(f"  samples per example      : {self.samples_per_example}")

def example_should_be_used(obstacles, occlusion, training_config):
    assert isinstance(obstacles, list)
    assert isinstance(occlusion, bool)
    assert isinstance(training_config, TrainingConfig)

    def allCircles(obs):
        for o in obs:
            if o[0] != CIRCLE:
                return False
        return True

    if not training_config.allow_occlusions and occlusion:
        return False
    if training_config.max_obstacles is not None and len(obstacles) > training_config.max_obstacles:
        return False
    if training_config.circles_only and not allCircles(obstacles):
        return False

    return True