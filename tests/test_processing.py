import numpy as np
import pytest

from processing import FilterSpec, design_and_apply
from analysis import compute_fft


def dominant_frequency(freqs: np.ndarray, mag: np.ndarray) -> float:
    if len(freqs) <= 1:
        return 0.0
    idx = np.argmax(mag[1:]) + 1
    return float(freqs[idx])


def test_none_returns_copy():
    x = np.array([1.0, 2.0, 3.0, 4.0])
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


def test_invalid_bpf_cutoffs_raise():
    x = np.ones(100)
    spec = FilterSpec(
        filter_type="BPF",
        method="FIR",
        order=21,
        cutoff1=20.0,
        cutoff2=10.0,
        fs=100.0,
    )
    with pytest.raises(ValueError):
        design_and_apply(x, spec)


def test_lpf_reduces_high_frequency_component():
    fs = 200.0
    t = np.arange(0, 2.0, 1 / fs)

    low = np.sin(2 * np.pi * 5 * t)
    high = 0.7 * np.sin(2 * np.pi * 50 * t)
    x = low + high

    spec = FilterSpec(
        filter_type="LPF",
        method="FIR",
        order=51,
        cutoff1=10.0,
        cutoff2=0.0,
        fs=fs,
    )

    y = design_and_apply(x, spec)

    freqs_x, mag_x = compute_fft(list(x), t, fs_hint=fs)
    freqs_y, mag_y = compute_fft(list(y), t, fs_hint=fs)

    dom_x = dominant_frequency(freqs_x, mag_x)
    dom_y = dominant_frequency(freqs_y, mag_y)

    assert dom_y < 20.0
    assert abs(dom_y - 5.0) < 2.0


def test_hpf_reduces_dc_and_low_frequency():
    fs = 200.0
    t = np.arange(0, 2.0, 1 / fs)

    slow = 2.0 * np.sin(2 * np.pi * 1 * t)
    fast = 0.8 * np.sin(2 * np.pi * 25 * t)
    x = slow + fast + 3.0  # DC offset

    spec = FilterSpec(
        filter_type="HPF",
        method="FIR",
        order=51,
        cutoff1=10.0,
        cutoff2=0.0,
        fs=fs,
    )

    y = design_and_apply(x, spec)

    assert abs(np.mean(y)) < 0.5

    freqs_y, mag_y = compute_fft(list(y), t, fs_hint=fs)
    dom_y = dominant_frequency(freqs_y, mag_y)

    assert abs(dom_y - 25.0) < 3.0


def test_bpf_keeps_mid_band():
    fs = 200.0
    t = np.arange(0, 2.0, 1 / fs)

    low = 0.8 * np.sin(2 * np.pi * 3 * t)
    mid = 1.0 * np.sin(2 * np.pi * 20 * t)
    high = 0.8 * np.sin(2 * np.pi * 60 * t)
    x = low + mid + high

    spec = FilterSpec(
        filter_type="BPF",
        method="FIR",
        order=51,
        cutoff1=10.0,
        cutoff2=30.0,
        fs=fs,
    )

    y = design_and_apply(x, spec)

    freqs_y, mag_y = compute_fft(list(y), t, fs_hint=fs)
    dom_y = dominant_frequency(freqs_y, mag_y)

    assert abs(dom_y - 20.0) < 3.0


def test_iir_bpf_runs():
    fs = 200.0
    t = np.arange(0, 2.0, 1 / fs)
    x = np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)

    spec = FilterSpec(
        filter_type="BPF",
        method="IIR",
        order=4,
        cutoff1=10.0,
        cutoff2=30.0,
        fs=fs,
    )

    y = design_and_apply(x, spec)

    assert len(y) == len(x)
    assert np.all(np.isfinite(y))