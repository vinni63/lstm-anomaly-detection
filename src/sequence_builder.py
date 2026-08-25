"""Convert a 1D series into (window -> next value) supervised examples."""
from __future__ import annotations
import numpy as np


def create_sequences(values: np.ndarray, window_size: int):
    """For [t1..t30,t31], return X=[t1..t30] and y=t31.

    Each following sample shifts forward by one point, which is the sliding window.
    Output X has Keras LSTM shape: (samples, timesteps, features=1).
    """
    values = np.asarray(values, dtype="float32").reshape(-1)
    if window_size < 1 or len(values) <= window_size:
        raise ValueError("Need more values than a positive window_size.")
    x = np.array([values[i : i + window_size] for i in range(len(values) - window_size)])
    y = values[window_size:]
    return x.reshape(-1, window_size, 1), y.reshape(-1, 1)

