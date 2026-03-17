import numpy as np
import pytest

from analysis import compute_fft
from processing import (
    FilterSpec,
    apply_fir,
    design_and_apply,
    design_fir_bpf,
    design_fir_lpf,
)


def dominant_frequency(freqs: np.ndarray, mag: np.ndarray) -> float:
    if len(freqs) <= 1:
        return 0.0
    idx = np.argmax(mag[1:]) + 1
    return float(freqs[idx])


def test_design_fir_lpf_length_and_sum():
    h = design_fir_lpf(fc=10.0, fs=100.0, numtaps=21)
    assert len(h) == 21
    assert np.isclose(np.sum(h), 1.0, atol=1e-6)


def test_design_fir_bpf_invalid_cutoffs():
    with pytest.raises(ValueError):
        design_fir_bpf(20.0, 10.0, 100.0, 21)


def test_apply_fir_same_length():
    x = np.ones(100)
    h = np.ones(5) / 5
    y = apply_fir(x, h)
    assert len(y) == len(x)


def test_design_and_apply_none_returns_copy():
    x = np.array([1.0, 2.0, 3.0])
    spec = FilterSpec(
        filter_type="None",
        method="FIR",
        order=11,
        cutoff1=10.0,
        cutoff2=20.0,
        fs=100.0,
    )
    y = design_and_apply(x, spec)
    assert np.allclose(x, y)
    assert y is not x


def test_lpf_reduces_high_frequency_component():
    fs = 200.0
    t = np.arange(0, 2.0, 1 / fs)
    x = np.sin(2 * np.pi * 5.0 * t) + 0.8 * np.sin(2 * np.pi * 50.0 * t)

    spec = FilterSpec(
        filter_type="LPF",
        method="FIR",
        order=31,
        cutoff1=10.0,
        cutoff2=0.0,
        fs=fs,
    )
    y = design_and_apply(x, spec)

    freqs_x, mag_x = compute_fft(x, t)
    freqs_y, mag_y = compute_fft(y, t)

    dom_after = dominant_frequency(freqs_y, mag_y)
    assert abs(dom_after - 5.0) < 2.0