import time
import numpy as np

from analysis import compute_fft
from processing import FilterSpec, design_and_apply


def dominant_frequency(freqs: np.ndarray, mag: np.ndarray) -> float:
    if len(freqs) <= 1:
        return 0.0
    idx = np.argmax(mag[1:]) + 1
    return float(freqs[idx])


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def stopband_energy(freqs: np.ndarray, mag: np.ndarray, filter_type: str, c1: float, c2: float = 0.0) -> float:
    power = mag ** 2
    ft = filter_type.upper()

    if ft == "LPF":
        mask = freqs > c1
    elif ft == "HPF":
        mask = freqs < c1
    elif ft == "BPF":
        low = min(c1, c2)
        high = max(c1, c2)
        mask = (freqs < low) | (freqs > high)
    else:
        mask = np.zeros_like(freqs, dtype=bool)

    return float(np.sum(power[mask]))


def evaluate_case(x: np.ndarray, t: np.ndarray, spec: FilterSpec):
    start = time.perf_counter()
    y = design_and_apply(x, spec)
    runtime_ms = (time.perf_counter() - start) * 1000.0

    freqs_x, mag_x = compute_fft(x, t)
    freqs_y, mag_y = compute_fft(y, t)

    raw_rms = rms(x)
    filt_rms = rms(y)
    rms_ratio = filt_rms / raw_rms if raw_rms != 0 else float("nan")

    raw_stop = stopband_energy(freqs_x, mag_x, spec.filter_type, spec.cutoff1, spec.cutoff2)
    filt_stop = stopband_energy(freqs_y, mag_y, spec.filter_type, spec.cutoff1, spec.cutoff2)
    stopband_change_pct = ((filt_stop - raw_stop) / raw_stop * 100.0) if raw_stop != 0 else float("nan")

    dom_raw = dominant_frequency(freqs_x, mag_x)
    dom_filt = dominant_frequency(freqs_y, mag_y)
    dom_change = abs(dom_filt - dom_raw)

    return {
        "runtime_ms": runtime_ms,
        "rms_ratio": rms_ratio,
        "stopband_change_pct": stopband_change_pct,
        "dom_freq_change_hz": dom_change,
    }


def build_signal(fs: float, duration: float):
    t = np.arange(0, duration, 1 / fs)
    x = (
        1.0 * np.sin(2 * np.pi * 5.0 * t)
        + 0.6 * np.sin(2 * np.pi * 25.0 * t)
        + 0.4 * np.sin(2 * np.pi * 60.0 * t)
        + 0.15 * np.random.randn(t.size)
    )
    return t, x


def main():
    fs = 250.0
    t, x = build_signal(fs, 8.0)

    cases = [
        FilterSpec("LPF", "FIR", 31, 10.0, 0.0, fs),
        FilterSpec("HPF", "FIR", 31, 10.0, 0.0, fs),
        FilterSpec("BPF", "FIR", 41, 4.0, 15.0, fs),
        FilterSpec("LPF", "IIR", 4, 10.0, 0.0, fs),
        FilterSpec("HPF", "IIR", 4, 10.0, 0.0, fs),
        FilterSpec("BPF", "IIR", 4, 4.0, 15.0, fs),
    ]

    print("=" * 88)
    print("WEEK 11 FILTER PERFORMANCE EVALUATION")
    print("=" * 88)

    for spec in cases:
        result = evaluate_case(x, t, spec)
        print(
            f"{spec.method:>3} {spec.filter_type:>3} | "
            f"order={spec.order:<3} | "
            f"RMS ratio={result['rms_ratio']:.4f} | "
            f"Stopband Δ={result['stopband_change_pct']:.2f}% | "
            f"Dom f Δ={result['dom_freq_change_hz']:.4f} Hz | "
            f"Runtime={result['runtime_ms']:.4f} ms"
        )

    print("\nPerformance test sizes:")
    for n in [1000, 5000, 10000, 50000]:
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * 5.0 * t) + 0.3 * np.random.randn(n)
        spec = FilterSpec("LPF", "FIR", 31, 10.0, 0.0, fs)

        start = time.perf_counter()
        _ = design_and_apply(x, spec)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        print(f"N={n:<6} runtime={runtime_ms:.4f} ms")


if __name__ == "__main__":
    main()