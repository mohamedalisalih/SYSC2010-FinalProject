import csv
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SignalData:
    t: Optional[List[float]]
    x: List[float]
    time_name: Optional[str] = None
    signal_name: Optional[str] = None


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _choose_signal_column(headers: List[str], preferred_signal_type: Optional[str] = None) -> int:
    normalized = [h.strip().lower() for h in headers]

    if preferred_signal_type:
        st = preferred_signal_type.strip().lower()

        if st == "respiration":
            for key in ["resp", "respiration"]:
                for i, h in enumerate(normalized):
                    if key == h or key in h:
                        return i

        elif st == "ecg":
            for key in ["ecg", "ii", "lead ii"]:
                for i, h in enumerate(normalized):
                    if key == h or key in h:
                        return i

        elif st == "temperature":
            for key in ["temp", "temperature"]:
                for i, h in enumerate(normalized):
                    if key == h or key in h:
                        return i

        elif st == "motion":
            for key in ["acc", "imu", "motion", "x", "y", "z"]:
                for i, h in enumerate(normalized):
                    if key == h or key in h:
                        return i

    for key in ["resp", "ecg", "ii", "temp", "temperature", "motion", "imu", "pleth"]:
        for i, h in enumerate(normalized):
            if key == h or key in h:
                return i

    return 1 if len(headers) > 1 else 0


def load_csv_numeric(path: str, preferred_signal_type: Optional[str] = None) -> SignalData:
    with open(path, "r", newline="") as f:
        reader = list(csv.reader(f))

    rows = [[c.strip() for c in row] for row in reader if row and any(c.strip() for c in row)]
    if not rows:
        raise ValueError("CSV file is empty.")

    first_row = rows[0]
    first_row_numeric = [_safe_float(c) for c in first_row]
    has_header = any(v is None for v in first_row_numeric)

    t_vals: List[float] = []
    x_vals: List[float] = []

    if has_header:
        headers = first_row
        data_rows = rows[1:]

        time_idx = None
        for i, h in enumerate(headers):
            h0 = h.strip().lower()
            if h0 in ["time", "time [s]", "t", "seconds", "sec"]:
                time_idx = i
                break

        signal_idx = _choose_signal_column(headers, preferred_signal_type)

        if time_idx == signal_idx:
            candidates = [i for i in range(len(headers)) if i != time_idx]
            if not candidates:
                raise ValueError("Could not identify a signal column.")
            signal_idx = candidates[0]

        for row in data_rows:
            if signal_idx >= len(row):
                continue

            x = _safe_float(row[signal_idx])
            if x is None:
                continue

            x_vals.append(x)

            if time_idx is not None and time_idx < len(row):
                t = _safe_float(row[time_idx])
                if t is not None:
                    t_vals.append(t)

        if not x_vals:
            raise ValueError("No numeric signal data found in selected column.")

        if len(t_vals) != len(x_vals):
            t_vals = []

        return SignalData(
            t=t_vals if t_vals else None,
            x=x_vals,
            time_name=headers[time_idx] if time_idx is not None else None,
            signal_name=headers[signal_idx],
        )

    for row in rows:
        nums = [_safe_float(c) for c in row if c != ""]
        nums = [v for v in nums if v is not None]

        if len(nums) >= 2:
            t_vals.append(nums[0])
            x_vals.append(nums[1])
        elif len(nums) == 1:
            x_vals.append(nums[0])

    if not x_vals:
        raise ValueError("No numeric data found in the CSV file.")

    return SignalData(
        t=t_vals if len(t_vals) == len(x_vals) else None,
        x=x_vals
    )