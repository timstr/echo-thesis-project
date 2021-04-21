import math
import numpy as np

from config import (
    emitter_arrangement_all,
    emitter_arrangement_mono,
    emitter_arrangement_stereo,
    emitter_arrangement_surround,
    receiver_arrangement_all,
    receiver_arrangement_flat,
    receiver_arrangement_grid,
)

wavesim_field_size = 512
wavesim_duration = 2048

# Measured using `measure_speed_of_sound.py` at a size of 2048
wavesim_speed_of_sound = 0.7804932396917319

wavesim_emitter_locations = [
    (wavesim_field_size - 8, wavesim_field_size // 2 - 200),  # 0 - ll
    (wavesim_field_size - 8, wavesim_field_size // 2 - 100),  # 1 - l
    (wavesim_field_size - 8, wavesim_field_size // 2),  # 2 - m
    (wavesim_field_size - 8, wavesim_field_size // 2 + 100),  # 3 - r
    (wavesim_field_size - 8, wavesim_field_size // 2 + 200),  # 4 - rr
]

wavesim_receiver_locations = [
    (wavesim_field_size - 8, wavesim_field_size // 2 - 150),  # 0 - bll
    (wavesim_field_size - 8, wavesim_field_size // 2 - 50),  # 1 - bl
    (wavesim_field_size - 8, wavesim_field_size // 2 + 50),  # 2 - br
    (wavesim_field_size - 8, wavesim_field_size // 2 + 150),  # 3 - brr
    (wavesim_field_size - 16, wavesim_field_size // 2 - 150),  # 4 - tll
    (wavesim_field_size - 16, wavesim_field_size // 2 - 50),  # 5 - tl
    (wavesim_field_size - 16, wavesim_field_size // 2 + 50),  # 6 - tr
    (wavesim_field_size - 16, wavesim_field_size // 2 + 150),  # 7 - trr
]


def make_emitter_indices(arrangement):
    assert arrangement in emitter_arrangement_all
    if arrangement == emitter_arrangement_mono:
        return [2]
    elif arrangement == emitter_arrangement_stereo:
        return [1, 3]
    elif arrangement == emitter_arrangement_surround:
        return [0, 1, 2, 3, 4]
    else:
        raise Exception("Unrecognized emitter layout")


def make_receiver_indices(num_receivers, arrangement):
    assert num_receivers in [1, 2, 4, 8]
    assert arrangement in receiver_arrangement_all

    # if flat arrangement, then at most 4 receivers
    assert arrangement != receiver_arrangement_flat or num_receivers <= 4

    # if grid arrangement, then at least 4 receivers
    assert arrangement != receiver_arrangement_grid or num_receivers >= 4

    if num_receivers == 1:
        assert arrangement == receiver_arrangement_flat
        return [1]
    elif num_receivers == 2:
        assert arrangement == receiver_arrangement_flat
        return [1, 2]
    elif num_receivers == 4 and arrangement == receiver_arrangement_flat:
        return [0, 1, 2, 3]
    elif num_receivers == 4 and arrangement == receiver_arrangement_grid:
        return [1, 2, 5, 6]
    elif num_receivers == 8:
        assert arrangement == receiver_arrangement_grid
        return [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        raise Exception("Unrecognized receiver layout")
