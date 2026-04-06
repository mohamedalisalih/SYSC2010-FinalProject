import numpy as np
from dataclasses import dataclass
from scipy.signal import butter, sosfiltfilt, firwin, filtfilt


@dataclass
class FilterSpec:
    filter_type: str   # "LPF", "HPF", "BPF", "None"
    method: str        # "FIR" or "IIR"
    order: int
    cutoff1: float     # Hz
    cutoff2: float     # Hz (used only for BPF)
    fs: float          # Hz


def _validate_spec(spec: FilterSpec) -> None:
    ft = spec.filter_type.strip().upper()
    method = spec.method.strip().upper()
    fs = float(spec.fs)
    order = int(spec.order)

    if fs <= 0:
        raise ValueError("Sampling rate fs must be > 0.")

    if method not in {"FIR", "IIR"}:
        raise ValueError("Method must be FIR or IIR.")

    if ft not in {"NONE", "LPF", "HPF", "BPF"}:
        raise ValueError("Filter type must be None, LPF, HPF, or BPF.")

    if ft == "NONE":
        return

    nyq = fs / 2.0

    if spec.cutoff1 <= 0 or spec.cutoff1 >= nyq:
        raise ValueError(f"Cutoff1 must be between 0 and {nyq:.6f} Hz.")

    if ft == "BPF":
        if spec.cutoff2 <= 0 or spec.cutoff2 >= nyq:
            raise ValueError(f"Cutoff2 must be between 0 and {nyq:.6f} Hz.")
        if spec.cutoff2 <= spec.cutoff1:
            raise ValueError("For BPF, cutoff2 must be greater than cutoff1.")

    if order < 1:
        raise ValueError("Filter order must be >= 1.")

    # Very important: keep IIR order small.
    if method == "IIR" and order > 8:
        raise ValueError("For IIR, use a small order like 2, 4, or 6.")


def _safe_filtfilt(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    filtfilt can fail on short signals if padlen is too large.
    This version uses a safe pad length.
    """
    x = np.asarray(x, dtype=float)

    if x.size < 8:
        raise ValueError("Signal is too short for filtering.")

    default_padlen = 3 * (max(len(a), len(b)) - 1)
    padlen = min(default_padlen, x.size - 1)

    if padlen < 1:
        raise ValueError("Signal is too short for filtering.")

    return filtfilt(b, a, x, padlen=padlen)


def _safe_sosfiltfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Stable zero-phase IIR filtering using second-order sections.
    """
    x = np.asarray(x, dtype=float)

    if x.size < 8:
        raise ValueError("Signal is too short for filtering.")

    # Let scipy choose padlen unless signal is extremely short.
    return sosfiltfilt(sos, x)


def design_and_apply(x: np.ndarray, spec: FilterSpec) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    _validate_spec(spec)

    ft = spec.filter_type.strip().upper()
    method = spec.method.strip().upper()
    order = int(spec.order)
    fs = float(spec.fs)

    if ft == "NONE":
        return x.copy()

    # FIR
    if method == "FIR":
        numtaps = max(5, order)
        if numtaps % 2 == 0:
            numtaps += 1  # odd length is better for linear-phase FIR

        if ft == "LPF":
            b = firwin(numtaps, spec.cutoff1, fs=fs, pass_zero="lowpass")
        elif ft == "HPF":
            b = firwin(numtaps, spec.cutoff1, fs=fs, pass_zero="highpass")
        elif ft == "BPF":
            b = firwin(numtaps, [spec.cutoff1, spec.cutoff2], fs=fs, pass_zero="bandpass")
        else:
            raise ValueError("Unknown FIR filter type.")

        return _safe_filtfilt(b, np.array([1.0]), x)

    # IIR
    if method == "IIR":
        if ft == "LPF":
            sos = butter(order, spec.cutoff1, btype="lowpass", fs=fs, output="sos")
        elif ft == "HPF":
            sos = butter(order, spec.cutoff1, btype="highpass", fs=fs, output="sos")
        elif ft == "BPF":
            sos = butter(order, [spec.cutoff1, spec.cutoff2], btype="bandpass", fs=fs, output="sos")
        else:
            raise ValueError("Unknown IIR filter type.")

        return _safe_sosfiltfilt(sos, x)

    raise ValueError("Method must be FIR or IIR.")