import numpy as np

from analysis import (
    compute_basic_metrics,
    compute_fft,
    estimate_temp_slope,
    make_time_axis,
)


def dominant_frequency(freqs: np.ndarray, mag: np.ndarray) -> float:
    if len(freqs) <= 1:
        return 0.0
    idx = np.argmax(mag[1:]) + 1
    return float(freqs[idx])


def test_make_time_axis_with_given_time():
    x = [1, 2, 3]
    t = [0.0, 0.1, 0.2]
    out = make_time_axis(t, x)
    assert np.allclose(out, [0.0, 0.1, 0.2])


def test_make_time_axis_fallbacks_to_index():
    x = [1, 2, 3, 4]
    out = make_time_axis(None, x)
    assert np.allclose(out, [0, 1, 2, 3])


def test_compute_basic_metrics():
    x = np.array([1.0, -1.0, 1.0, -1.0])
    m = compute_basic_metrics(x)
    assert abs(m["mean"] - 0.0) < 1e-12
    assert abs(m["rms"] - 1.0) < 1e-12
    assert abs(m["ptp"] - 2.0) < 1e-12


def test_fft_detects_main_frequency():
    fs = 100.0
    t = np.arange(0, 2.0, 1 / fs)
    x = np.sin(2 * np.pi * 5.0 * t)

    freqs, mag = compute_fft(x, t)
    dom = dominant_frequency(freqs, mag)
    assert abs(dom - 5.0) < 0.5


def test_estimate_temp_slope():
    t = np.array([0, 1, 2, 3, 4], dtype=float)
    x = 2.0 * t + 1.0
    slope = estimate_temp_slope(t, x)
    assert slope is not None
    assert abs(slope - 2.0) < 1e-12