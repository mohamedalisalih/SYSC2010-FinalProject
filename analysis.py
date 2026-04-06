import numpy as np
from typing import Tuple, Optional, Dict


def make_time_axis(t: Optional[list[float]], x: list[float]) -> np.ndarray:
    n = len(x)
    if t is not None and len(t) == n:
        return np.array(t, dtype=float)
    return np.arange(n, dtype=float)


def estimate_fs_from_t(t_axis: np.ndarray) -> Optional[float]:
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


def compute_fft(x: list[float], t_axis: np.ndarray, fs_hint: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    x_np = np.array(x, dtype=float)
    n = x_np.size
    if n < 2:
        return np.array([]), np.array([])

    x_np = x_np - np.mean(x_np)

    fs = fs_hint if fs_hint is not None and fs_hint > 0 else estimate_fs_from_t(t_axis)
    if fs is None:
        fs = 1.0

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(x_np)) / n
    return freqs, mag


def compute_basic_metrics(x: np.ndarray) -> Dict[str, float]:
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
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return np.array([], dtype=int)

    candidates = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > threshold))[0] + 1
    if candidates.size == 0:
        return np.array([], dtype=int)

    peaks = []
    last_keep = -10**9

    for idx in candidates:
        if idx - last_keep >= min_distance:
            peaks.append(idx)
            last_keep = idx
        else:
            if x[idx] > x[peaks[-1]]:
                peaks[-1] = idx
                last_keep = idx

    return np.array(peaks, dtype=int)


def estimate_hr_ecg(t_axis: np.ndarray, x: np.ndarray, fs_hint: Optional[float] = None) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    t_axis = np.asarray(t_axis, dtype=float)
    if x.size < 5:
        return None

    fs = estimate_fs_from_t(t_axis) or fs_hint
    if fs is None or fs <= 0:
        return None

    x0 = x - np.mean(x)
    s = np.std(x0)
    if s > 0:
        x0 = x0 / s

    min_distance = int(0.25 * fs)
    threshold = 0.8
    peaks = _simple_peaks(x0, min_distance=min_distance, threshold=threshold)

    if peaks.size < 2:
        return None

    t_peaks = t_axis[peaks]
    rr = np.diff(t_peaks)
    rr = rr[(rr > 0.25) & (rr < 2.0)]
    if rr.size == 0:
        return None

    return float(60.0 / np.median(rr))


def estimate_resp_rate(t_axis: np.ndarray, x: np.ndarray, fs_hint: Optional[float] = None) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    t_axis = np.asarray(t_axis, dtype=float)
    if x.size < 5:
        return None

    fs = estimate_fs_from_t(t_axis) or fs_hint
    if fs is None or fs <= 0:
        return None

    x0 = x - np.mean(x)
    s = np.std(x0)
    if s > 0:
        x0 = x0 / s

    min_distance = int(1.0 * fs)
    threshold = 0.2
    peaks = _simple_peaks(x0, min_distance=min_distance, threshold=threshold)

    if peaks.size < 2:
        return None

    t_peaks = t_axis[peaks]
    periods = np.diff(t_peaks)
    periods = periods[(periods > 1.0) & (periods < 10.0)]
    if periods.size == 0:
        return None

    return float(60.0 / np.median(periods))


def estimate_temp_slope(t_axis: np.ndarray, x: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    t_axis = np.asarray(t_axis, dtype=float)

    if x.size < 2:
        return None

    if not np.all(np.isfinite(t_axis)) or not np.all(np.isfinite(x)):
        return None

    A = np.vstack([t_axis, np.ones_like(t_axis)]).T
    coeff, *_ = np.linalg.lstsq(A, x, rcond=None)
    return float(coeff[0])