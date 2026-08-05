"""Layout A: self-describing, and the only interface between the two parts."""

from __future__ import annotations

import numpy as np
import pytest

from polyrmc.io_csv import COLUMNS, read_processed_csv, write_processed_csv

SOURCE_HEADER = [
    "Header Rows: 4",
    "Experiment: liraglutide stability",
    "Sample Concentration: 1.5 mg/mL",
]


def _columns(n: int = 50) -> dict[str, np.ndarray]:
    seconds = np.arange(n, dtype=float)
    return {
        "seconds": seconds,
        "concentration": 1e-3 * np.exp(-seconds / 100.0),
        "ls8": 1000.0 - seconds,
        "ls8_smoothed": 1000.0 - seconds,
        "I_rayleigh": np.full(n, 1e-5),
        "Mw_over_M0": np.ones(n),
    }


def test_the_schema_is_the_six_design_columns_in_dependency_order():
    assert list(COLUMNS) == [
        "seconds",
        "concentration",
        "ls8",
        "ls8_smoothed",
        "I_rayleigh",
        "Mw_over_M0",
    ]


def test_roundtrip_preserves_values_and_metadata(tmp_path):
    path = write_processed_csv(
        tmp_path / "run.csv", _columns(), metadata={"run_id": "abc", "channel": "ls8"}
    )
    frame, metadata = read_processed_csv(path)

    assert list(frame.columns) == list(COLUMNS)
    assert len(frame) == 50
    assert metadata["run_id"] == "abc"
    np.testing.assert_allclose(frame["seconds"], _columns()["seconds"])


def test_the_source_header_is_carried_verbatim(tmp_path):
    """An output that carries its own origin can be read without a second file."""
    path = write_processed_csv(
        tmp_path / "run.csv", _columns(), source_header=SOURCE_HEADER
    )
    lines = path.read_text(encoding="utf-8").splitlines()

    assert "Experiment: liraglutide stability" in lines
    assert "Sample Concentration: 1.5 mg/mL" in lines


def test_the_declared_length_is_corrected_not_copied(tmp_path):
    """The source says 4 rows. This file is longer, and must say so."""
    path = write_processed_csv(
        tmp_path / "run.csv",
        _columns(),
        metadata={"run_id": "abc"},
        source_header=SOURCE_HEADER,
    )
    lines = path.read_text(encoding="utf-8").splitlines()

    # 1 declaration + 2 carried + 1 pipeline + 1 column row.
    assert lines[0] == "Header Rows: 5"
    assert lines[4].startswith("seconds,")
    assert "Header Rows: 4" not in lines[1:]


def test_the_declared_length_lands_on_the_column_row(tmp_path):
    """The count includes the column row, as the instrument's own convention does."""
    for n_metadata in (1, 12):
        path = write_processed_csv(
            tmp_path / f"run{n_metadata}.csv",
            _columns(),
            metadata={f"key_{i}": str(i) for i in range(n_metadata)},
            source_header=SOURCE_HEADER,
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        declared = int(lines[0].split(":")[1])
        assert lines[declared - 1].startswith("seconds,")


def test_reader_uses_the_declared_length_regardless_of_header_size(tmp_path):
    small = write_processed_csv(tmp_path / "small.csv", _columns(), metadata={"a": "1"})
    large = write_processed_csv(
        tmp_path / "large.csv",
        _columns(),
        metadata={f"key_{i}": str(i) for i in range(12)},
        source_header=SOURCE_HEADER,
    )

    for path in (small, large):
        frame, _metadata = read_processed_csv(path)
        assert len(frame) == 50
        assert list(frame.columns) == list(COLUMNS)


def test_a_non_ls8_channel_names_its_own_columns(tmp_path):
    path = write_processed_csv(
        tmp_path / "run.csv",
        _columns(),
        metadata={"channel": "Cell_8"},
        channel="Cell_8",
    )
    header = path.read_text(encoding="utf-8").splitlines()[2]
    assert "Cell_8" in header and "Cell_8_smoothed" in header

    # ...and reads back under the canonical names.
    frame, _metadata = read_processed_csv(path)
    assert list(frame.columns) == list(COLUMNS)


def test_files_written_under_the_old_column_names_still_read(tmp_path):
    """Reprocessing a run to read its own output would defeat a stable boundary."""
    path = tmp_path / "legacy.csv"
    path.write_text(
        "Header Rows: 3\n"
        "run_id: old\n"
        "time_s,concentration_g_per_cm3,signal_raw,signal_smoothed,"
        "rayleigh_ratio_cm_inv,mw_over_m0\n"
        "0,1e-3,1000,1000,1e-5,1\n"
        "1,1e-3,999,999,1e-5,1\n",
        encoding="utf-8",
    )
    frame, metadata = read_processed_csv(path)

    assert list(frame.columns) == list(COLUMNS)
    assert metadata["run_id"] == "old"
    assert len(frame) == 2


def test_missing_column_is_refused_at_write_time(tmp_path):
    columns = _columns()
    del columns["Mw_over_M0"]

    with pytest.raises(ValueError, match="missing required columns"):
        write_processed_csv(tmp_path / "bad.csv", columns)


def test_a_file_without_a_declaration_is_refused_at_read_time(tmp_path):
    path = tmp_path / "foreign.csv"
    path.write_text("seconds,concentration\n0,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not declare its header length"):
        read_processed_csv(path)
