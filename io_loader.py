import csv
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SignalData:
    t: Optional[List[float]]
    x: List[float]


def load_csv_numeric(path: str) -> SignalData:
    t_vals: List[float] = []
    x_vals: List[float] = []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            row = [c.strip() for c in row]
            if all(c == "" for c in row):
                continue

            try:
                nums = [float(c) for c in row if c != ""]
            except ValueError:
                # header / non-numeric row
                continue

            if len(nums) == 0:
                continue

            if len(nums) >= 2:
                t_vals.append(nums[0])
                x_vals.append(nums[1])
            else:
                x_vals.append(nums[0])

    if len(x_vals) == 0:
        raise ValueError("No numeric data found in the CSV file.")

    if len(t_vals) == 0:
        return SignalData(t=None, x=x_vals)

    return SignalData(t=t_vals, x=x_vals)