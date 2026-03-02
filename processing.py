import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FilterSpec:
    filter_type: str   # "LPF", "HPF", "BPF", "None"
    method: str        # "FIR" or "IIR"
    order: int
    cutoff1: float     # Hz
    cutoff2: float     # Hz (only for BPF)
    fs: float          # Hz


def _sinc(x: np.ndarray) -> np.ndarray:
    # np.sinc uses sin(pi x)/(pi x) already
    return np.sinc(x)


def design_fir_lpf(fc: float, fs: float, numtaps: int) -> np.ndarray:
    # Windowed-sinc lowpass (Hamming)
    n = np.arange(numtaps)
    M = numtaps - 1
    h = 2 * fc / fs * _sinc(2 * fc / fs * (n - M / 2))
    w = np.hamming(numtaps)
    h = h * w
    h = h / np.sum(h)  # normalize DC gain
    return h


def design_fir_hpf(fc: float, fs: float, numtaps: int) -> np.ndarray:
    # Spectral inversion of LPF
    h_lpf = design_fir_lpf(fc, fs, numtaps)
    h = -h_lpf
    h[numtaps // 2] += 1.0
    return h


def design_fir_bpf(f1: float, f2: float, fs: float, numtaps: int) -> np.ndarray:
    if f2 <= f1:
        raise ValueError("For BPF, cutoff2 must be > cutoff1.")
    h2 = design_fir_lpf(f2, fs, numtaps)
    h1 = design_fir_lpf(f1, fs, numtaps)
    return h2 - h1


def design_iir_butter_lowpass(fc: float, fs: float, order: int) -> Tuple[np.ndarray, np.ndarray]:
    # Simple Butterworth via bilinear transform of analog prototype
    # We’ll use scipy-like math but implemented with numpy only (biquad cascade approach is more robust;
    # for course-level demo, we’ll implement a single-stage if order=2, and cascade if higher even order).
    return _butter_iir(fc, fs, order, ftype="low")


def design_iir_butter_highpass(fc: float, fs: float, order: int) -> Tuple[np.ndarray, np.ndarray]:
    return _butter_iir(fc, fs, order, ftype="high")


def design_iir_butter_bandpass(f1: float, f2: float, fs: float, order: int) -> Tuple[np.ndarray, np.ndarray]:
    if f2 <= f1:
        raise ValueError("For BPF, cutoff2 must be > cutoff1.")
    # Bandpass as cascade: HPF at f1 then LPF at f2 (not a true Butter bandpass, but works well enough for the project)
    b_hp, a_hp = _butter_iir(f1, fs, order, ftype="high")
    b_lp, a_lp = _butter_iir(f2, fs, order, ftype="low")
    # Combine by convolution of polynomials (equivalent to cascade)
    b = np.convolve(b_hp, b_lp)
    a = np.convolve(a_hp, a_lp)
    return b, a


def _butter_iir(fc: float, fs: float, order: int, ftype: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    A practical “good enough” digital Butterworth using bilinear transform and pole placement.
    Implements an Nth order low/highpass by cascading 2nd-order sections (SOS) but returns
    combined (b, a) polynomials (OK for moderate orders).
    """
    if order < 1:
        raise ValueError("IIR order must be >= 1.")
    if fc <= 0 or fc >= fs / 2:
        raise ValueError("Cutoff must be between 0 and Nyquist (fs/2).")

    # Prewarp
    wc = 2 * fs * np.tan(np.pi * fc / fs)

    # Analog Butterworth poles on unit circle in s-plane
    poles = []
    for k in range(order):
        theta = np.pi * (2 * k + 1 + order) / (2 * order)
        poles.append(wc * np.exp(1j * theta))
    poles = np.array(poles)

    # Lowpass prototype transfer function:
    # H(s) = wc^N / Π(s - p_k)
    # Bilinear transform: s = 2fs (1 - z^-1)/(1 + z^-1)

    # Convert analog poles to digital poles via bilinear transform
    # z = (2fs + s) / (2fs - s)
    z_poles = (2 * fs + poles) / (2 * fs - poles)

    # Zeros:
    if ftype == "low":
        z_zeros = np.array([-1.0] * order)  # zeros at z = -1 (mapped from infinity)
        gain = np.real(np.prod(2 * fs - poles) / np.prod(2 * fs + poles))
    elif ftype == "high":
        z_zeros = np.array([1.0] * order)   # zeros at z = +1
        # For highpass, frequency transform: s -> wc^2 / s
        # Instead of full transform, we use lowpass and spectral inversion-ish mapping by swapping z=-1 to z=+1
        # and adjusting gain by evaluating at z=-1 (Nyquist) vs z=1 (DC).
        gain = np.real(np.prod(-(2 * fs - poles)) / np.prod(2 * fs + poles))
    else:
        raise ValueError("ftype must be 'low' or 'high'.")

    # Form polynomials from zeros/poles
    b = gain * np.poly(z_zeros).real
    a = np.poly(z_poles).real

    # Normalize gain:
    if ftype == "low":
        # Normalize DC gain to 1
        z = 1.0
    else:
        # Normalize gain at Nyquist to 1
        z = -1.0
    H = np.polyval(b, z) / np.polyval(a, z)
    b = b / H.real

    return b, a


def apply_fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    # Same length output using 'same' convolution
    return np.convolve(x, h, mode="same")


def apply_iir(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    # Direct Form I filtering (simple). For stability, keep orders moderate.
    y = np.zeros_like(x, dtype=float)
    for n in range(len(x)):
        acc = 0.0
        for i in range(len(b)):
            if n - i >= 0:
                acc += b[i] * x[n - i]
        for j in range(1, len(a)):
            if n - j >= 0:
                acc -= a[j] * y[n - j]
        y[n] = acc / a[0]
    return y


def design_and_apply(x: np.ndarray, spec: FilterSpec) -> np.ndarray:
    ft = spec.filter_type
    if ft == "None":
        return x.copy()

    method = spec.method
    order = int(spec.order)
    fs = float(spec.fs)

    if fs <= 0:
        raise ValueError("Sampling rate fs must be > 0.")

    if method == "FIR":
        # FIR uses numtaps; force odd for nicer symmetry
        numtaps = max(5, order)
        if numtaps % 2 == 0:
            numtaps += 1

        if ft == "LPF":
            h = design_fir_lpf(spec.cutoff1, fs, numtaps)
        elif ft == "HPF":
            h = design_fir_hpf(spec.cutoff1, fs, numtaps)
        elif ft == "BPF":
            h = design_fir_bpf(spec.cutoff1, spec.cutoff2, fs, numtaps)
        else:
            raise ValueError("Unknown filter type.")
        return apply_fir(x, h)

    elif method == "IIR":
        if ft == "LPF":
            b, a = design_iir_butter_lowpass(spec.cutoff1, fs, order)
        elif ft == "HPF":
            b, a = design_iir_butter_highpass(spec.cutoff1, fs, order)
        elif ft == "BPF":
            b, a = design_iir_butter_bandpass(spec.cutoff1, spec.cutoff2, fs, order)
        else:
            raise ValueError("Unknown filter type.")
        return apply_iir(x, b, a)

    else:
        raise ValueError("Method must be FIR or IIR.")