"""The CSV boundary: self-describing, and the only interface between parts."""

from __future__ import annotations

import numpy as np
import pytest

from polyrmc.io_csv import COLUMNS, read_processed_csv, write_processed_csv


def _columns(n: int = 50) -> dict[str, np.ndarray]:
    time_s = np.arange(n, dtype=float)
    return {
        "time_s": time_s,
        "concentration_g_per_cm3": 1e-3 * np.exp(-time_s / 100.0),
        "signal_raw": 1000.0 - time_s,
        "signal_smoothed": 1000.0 - time_s,
        "rayleigh_ratio_cm_inv": np.full(n, 1e-5),
        "mw_over_m0": np.ones(n),
    }


def test_the_schema_is_six_columns_in_dependency_order():
    assert len(COLUMNS) == 6
    assert COLUMNS[0] == "time_s"
    assert COLUMNS[1] == "concentration_g_per_cm3"


def test_roundtrip_preserves_values_and_metadata(tmp_path):
    path = write_processed_csv(
        tmp_path / "run.csv", _columns(), metadata={"run_id": "abc", "channel": "Cell_1"}
    )
    frame, metadata = read_processed_csv(path)

    assert list(frame.columns) == list(COLUMNS)
    assert len(frame) == 50
    assert metadata["run_id"] == "abc"
    assert metadata["channel"] == "Cell_1"
    np.testing.assert_allclose(frame["time_s"], _columns()["time_s"])


def test_header_length_is_declared_on_the_first_line(tmp_path):
    path = write_processed_csv(tmp_path / "run.csv", _columns(), metadata={"a": "1"})
    first_line = path.read_text(encoding="utf-8").splitlines()[0]

    assert "Number of header rows" in first_line


def test_reader_uses_the_declared_length_regardless_of_metadata_size(tmp_path):
    """Part 2 re-reads with the same convention Part 1 used -- no fixed skiprows."""
    small = write_processed_csv(tmp_path / "small.csv", _columns(), metadata={"a": "1"})
    large = write_processed_csv(
        tmp_path / "large.csv",
        _columns(),
        metadata={f"key_{i}": str(i) for i in range(12)},
    )

    for path in (small, large):
        frame, _metadata = read_processed_csv(path)
        assert len(frame) == 50
        assert list(frame.columns) == list(COLUMNS)


def test_missing_column_is_refused_at_write_time(tmp_path):
    columns = _columns()
    del columns["mw_over_m0"]

    with pytest.raises(ValueError, match="missing required columns"):
        write_processed_csv(tmp_path / "bad.csv", columns)


def test_a_file_without_a_declaration_is_refused_at_read_time(tmp_path):
    path = tmp_path / "foreign.csv"
    path.write_text("time_s,concentration_g_per_cm3\n0,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not declare its header length"):
        read_processed_csv(path)
