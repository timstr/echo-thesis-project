import numpy as np
import torch

from wavesim_params import wavesim_duration
from config import (
    EmitterConfig,
    ReceiverConfig,
    InputConfig,
)
from config_constants import (
    input_format_audioraw,
    input_format_audiowaveshaped,
    input_format_spectrogram,
    input_format_gcc,
    input_format_gccphat,
)
from featurize_audio import sclog, make_spectrogram, make_gcc


def combine_emitted_signals(impulse_responses, emitter_config, receiver_config):
    assert isinstance(impulse_responses, np.ndarray) or isinstance(
        impulse_responses, torch.Tensor
    )
    assert impulse_responses.shape == (5, 8, wavesim_duration)
    assert isinstance(emitter_config, EmitterConfig)
    assert isinstance(receiver_config, ReceiverConfig)

    out_signal = np.zeros((receiver_config.count, wavesim_duration))

    assert len(emitter_config.indices) == len(emitter_config.emitted_signals)

    for ei, es in zip(emitter_config.indices, emitter_config.emitted_signals):
        assert es.shape == (wavesim_duration,)
        for ro, ri in enumerate(receiver_config.indices):
            conved = np.convolve(impulse_responses[ei][ri], es)[:wavesim_duration]
            assert conved.shape == (wavesim_duration,)
            out_signal[ro] += conved

    return out_signal


def transform_received_signals(received_signals, input_config):
    assert isinstance(input_config, InputConfig)
    assert isinstance(received_signals, np.ndarray) or isinstance(
        received_signals, torch.Tensor
    )
    assert len(received_signals.shape) == 2
    assert received_signals.shape[1] == wavesim_duration

    received_signals = torch.tensor(received_signals, dtype=torch.float)

    if input_config.format == input_format_audioraw:
        return received_signals
    elif input_config.format == input_format_audiowaveshaped:
        return sclog(received_signals)
    elif input_config.format == input_format_spectrogram:
        return make_spectrogram(received_signals)
    elif input_config.format == input_format_gcc:
        return make_gcc(input_config.emitted_signal, received_signals, transform=None)
    elif input_config.format == input_format_gccphat:
        return make_gcc(input_config.emitted_signal, received_signals, transform="phat")
    else:
        raise Exception(f"Unrecognized input format: '{input_config.format}'")
