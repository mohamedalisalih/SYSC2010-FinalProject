from pathlib import Path

import pytest

from io_loader import load_csv_numeric


def test_load_csv_two_columns_with_header(tmp_path: Path):
    p = tmp_path / "sig.csv"
    p.write_text("time,amp\n0.0,1.0\n0.1,2.0\n0.2,3.0\n", encoding="utf-8")

    data = load_csv_numeric(str(p))
    assert data.t == [0.0, 0.1, 0.2]
    assert data.x == [1.0, 2.0, 3.0]


def test_load_csv_single_column(tmp_path: Path):
    p = tmp_path / "single.csv"
    p.write_text("1\n2\n3\n4\n", encoding="utf-8")

    data = load_csv_numeric(str(p))
    assert data.t is None
    assert data.x == [1.0, 2.0, 3.0, 4.0]


def test_load_csv_skips_non_numeric_rows(tmp_path: Path):
    p = tmp_path / "mixed.csv"
    p.write_text("time,amp\nhello,world\n0.0,4.0\n0.1,5.0\n", encoding="utf-8")

    data = load_csv_numeric(str(p))
    assert data.t == [0.0, 0.1]
    assert data.x == [4.0, 5.0]


def test_load_csv_raises_on_no_numeric_data(tmp_path: Path):
    p = tmp_path / "empty.csv"
    p.write_text("a,b\nc,d\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_csv_numeric(str(p))