import numpy as np


def fill_missing_values(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float).copy()

    if x.size == 0:
        return x

    invalid = ~np.isfinite(x)
    if not np.any(invalid):
        return x

    valid_idx = np.where(~invalid)[0]
    if valid_idx.size == 0:
        raise ValueError("Signal contains only invalid values.")

    x[invalid] = np.interp(np.where(invalid)[0], valid_idx, x[valid_idx])
    return x


def remove_mean(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    if x.size == 0:
        return x.copy()
    return x - np.mean(x)


def detrend_linear(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    if x.size < 2:
        return x.copy()

    t = np.arange(x.size, dtype=float)
    coeffs = np.polyfit(t, x, 1)
    trend = np.polyval(coeffs, t)
    return x - trend


def normalize_zscore(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float)

    if x.size == 0:
        return x.copy()

    mean = np.mean(x)
    std = np.std(x)

    if std == 0:
        return x - mean

    return (x - mean) / std


def normalize_minmax(signal: np.ndarray, out_min: float = -1.0, out_max: float = 1.0) -> np.ndarray:
    x = np.asarray(signal, dtype=float)

    if x.size == 0:
        return x.copy()

    xmin = np.min(x)
    xmax = np.max(x)

    if xmax == xmin:
        return np.zeros_like(x)

    return (x - xmin) / (xmax - xmin) * (out_max - out_min) + out_min


def preprocess_signal(
    signal: np.ndarray,
    fill_missing: bool = True,
    baseline_method: str = "detrend",
    normalize_method: str = "zscore",
) -> np.ndarray:
    x = np.asarray(signal, dtype=float).copy()

    if fill_missing:
        x = fill_missing_values(x)

    if baseline_method == "none":
        pass
    elif baseline_method == "mean":
        x = remove_mean(x)
    elif baseline_method == "detrend":
        x = detrend_linear(x)
    else:
        raise ValueError(f"Unknown baseline_method: {baseline_method}")

    if normalize_method == "none":
        pass
    elif normalize_method == "zscore":
        x = normalize_zscore(x)
    elif normalize_method == "minmax":
        x = normalize_minmax(x)
    else:
        raise ValueError(f"Unknown normalize_method: {normalize_method}")

    return x


def get_preprocess_defaults(signal_type: str) -> dict:
    s = signal_type.strip().lower()

    if s == "ecg":
        return {
            "fill_missing": True,
            "baseline_method": "detrend",
            "normalize_method": "zscore",
        }

    if s == "respiration":
        return {
            "fill_missing": True,
            "baseline_method": "detrend",
            "normalize_method": "zscore",
        }

    if s == "temperature":
        return {
            "fill_missing": True,
            "baseline_method": "none",
            "normalize_method": "none",
        }

    if s == "motion":
        return {
            "fill_missing": True,
            "baseline_method": "mean",
            "normalize_method": "zscore",
        }

    return {
        "fill_missing": True,
        "baseline_method": "mean",
        "normalize_method": "zscore",
    }