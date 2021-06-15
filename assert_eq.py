import math
import numpy as np


def assert_eq(actual, expected):
    if isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise Exception(
                f"Expected something close to\n    {expected}, but got\n    {actual} instead"
            )
    elif isinstance(expected, np.ndarray):
        if not np.allclose(actual, expected):
            raise Exception(
                f"Expected something close to:\n    {expected}\nbut got\n    {actual}\ninstead"
            )
    elif not (actual == expected):
        raise Exception(f"Expected\n    {expected}, but got\n    {actual} instead")
