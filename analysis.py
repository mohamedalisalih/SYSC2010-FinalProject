import numpy as np
from typing import Tuple, Optional, Dict


def make_time_axis(t: Optional[list[float]], x: list[float]) -> np.ndarray:
    """
    If t exists and matches x length, return it.
    Otherwise return 0..N-1 as sample index.
    """
    n = len(x)
    if t is not None and len(t) == n:
        return np.array(t, dtype=float)
    return np.arange(n, dtype=float)


def _estimate_fs_from_t(t_axis: np.ndarray) -> Optional[float]:
    """
    Estimate sampling frequency from time axis. Returns None if not possible.
    """
    if t_axis.size < 2:
        return None
    diffs = np.diff(t_axis)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    dt = float(np.median(diffs))
    if dt <= 0:
        return None
    return 1.0 / dt


def compute_fft(x: list[float], t_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    rFFT magnitude. Estimates fs from time axis if possible.
    Returns (freqs, mag).
    """
    x_np = np.array(x, dtype=float)
    n = x_np.size
    if n < 2:
        return np.array([]), np.array([])

    # remove DC
    x_np = x_np - np.mean(x_np)

    fs = _estimate_fs_from_t(t_axis)
    if fs is None:
        fs = 1.0  # fallback

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(x_np)) / n
    return freqs, mag

# Week 10: metrics/features

def compute_basic_metrics(x: np.ndarray) -> Dict[str, float]:
    """
    Basic stats for a signal.
    Returns: mean, std, rms, ptp, min, max
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"mean": np.nan, "std": np.nan, "rms": np.nan, "ptp": np.nan, "min": np.nan, "max": np.nan}

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    rms = float(np.sqrt(np.mean(x ** 2)))
    ptp = float(np.ptp(x))
    mn = float(np.min(x))
    mx = float(np.max(x))
    return {"mean": mean, "std": std, "rms": rms, "ptp": ptp, "min": mn, "max": mx}


def _simple_peaks(x: np.ndarray, min_distance: int = 10, threshold: float = 0.0) -> np.ndarray:
    """
    Very simple peak detector:
    - local maxima
    - above threshold
    - enforces minimum distance between peaks
    Returns peak indices.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return np.array([], dtype=int)

    # local maxima
    candidates = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > threshold))[0] + 1
    if candidates.size == 0:
        return np.array([], dtype=int)

    # enforce min_distance by greedy selection (keep biggest peak in neighborhood)
    peaks = []
    last_keep = -10**9
    for idx in candidates:
        if idx - last_keep >= min_distance:
            peaks.append(idx)
            last_keep = idx
        else:
            # if too close, keep the taller one
            if x[idx] > x[peaks[-1]]:
                peaks[-1] = idx
                last_keep = idx

    return np.array(peaks, dtype=int)


def estimate_hr_ecg(t_axis: np.ndarray, x: np.ndarray, fs_hint: Optional[float] = None) -> Optional[float]:
    """
    Rough HR estimate using peak detection on a normalized ECG-like signal.
    Returns bpm or None if not enough peaks.
    """
    x = np.asarray(x, dtype=float)
    t_axis = np.asarray(t_axis, dtype=float)
    if x.size < 5:
        return None

    fs = _estimate_fs_from_t(t_axis) or fs_hint
    if fs is None or fs <= 0:
        return None

    # normalize
    x0 = x - np.mean(x)
    s = np.std(x0)
    if s > 0:
        x0 = x0 / s

    # ECG peaks: use threshold and refractory
    min_distance = int(0.25 * fs)  # 250ms
    threshold = 0.8               # z-score-ish
    peaks = _simple_peaks(x0, min_distance=min_distance, threshold=threshold)

    if peaks.size < 2:
        return None

    # compute RR intervals using time axis
    t_peaks = t_axis[peaks]
    rr = np.diff(t_peaks)
    rr = rr[(rr > 0.25) & (rr < 2.0)]  # keep realistic RR (30–240 bpm)
    if rr.size == 0:
        return None

    hr = 60.0 / float(np.median(rr))
    return float(hr)


def estimate_resp_rate(t_axis: np.ndarray, x: np.ndarray, fs_hint: Optional[float] = None) -> Optional[float]:
    """
    Rough breathing rate using peaks in a respiration-like waveform.
    Returns breaths/min or None.
    """
    x = np.asarray(x, dtype=float)
    t_axis = np.asarray(t_axis, dtype=float)
    if x.size < 5:
        return None

    fs = _estimate_fs_from_t(t_axis) or fs_hint
    if fs is None or fs <= 0:
        return None

    # normalize
    x0 = x - np.mean(x)
    s = np.std(x0)
    if s > 0:
        x0 = x0 / s

    # Resp is slow: enforce larger distance
    min_distance = int(1.0 * fs)  # 1s
    threshold = 0.2
    peaks = _simple_peaks(x0, min_distance=min_distance, threshold=threshold)

    if peaks.size < 2:
        return None

    t_peaks = t_axis[peaks]
    periods = np.diff(t_peaks)
    periods = periods[(periods > 1.0) & (periods < 10.0)]  # 6–60 breaths/min typical window
    if periods.size == 0:
        return None

    br = 60.0 / float(np.median(periods))
    return float(br)


def estimate_temp_slope(t_axis: np.ndarray, x: np.ndarray) -> Optional[float]:
    """
    Trend slope using linear fit: x ≈ a*t + b
    Returns slope a (units per second) or None.
    """
    x = np.asarray(x, dtype=float)
    t_axis = np.asarray(t_axis, dtype=float)
    if x.size < 2:
        return None

    # If time axis is just sample index, slope becomes units/sample (still okay)
    # but prefer real seconds if available.
    if not np.all(np.isfinite(t_axis)) or not np.all(np.isfinite(x)):
        return None

    # linear regression
    A = np.vstack([t_axis, np.ones_like(t_axis)]).T
    coeff, *_ = np.linalg.lstsq(A, x, rcond=None)
    slope = float(coeff[0])
    return slope