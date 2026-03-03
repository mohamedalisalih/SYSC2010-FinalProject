import numpy as np
from typing import Tuple, Optional


def make_time_axis(t: Optional[list[float]], x: list[float]) -> np.ndarray:
    """
    If t exists and matches x length, return it.
    Otherwise return 0..N-1.
    """
    n = len(x)
    if t is not None and len(t) == n:
        return np.array(t, dtype=float)
    return np.arange(n, dtype=float)


def compute_fft(x: list[float], t_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    rFFT magnitude. Estimates fs from time axis if possible.
    Returns (freqs, mag).
    """
    x_np = np.array(x, dtype=float)
    n = x_np.size

    if n < 2:
        return np.array([]), np.array([])

    # Remove DC offset
    x_np = x_np - np.mean(x_np)

    # Estimate sampling frequency
    dt = None
    if t_axis.size == n:
        diffs = np.diff(t_axis)
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size > 0 and np.all(diffs > 0):
            dt = float(np.median(diffs))

    fs = 1.0 if (dt is None or dt <= 0) else 1.0 / dt

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(x_np)) / n
    return freqs, mag