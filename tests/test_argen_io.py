"""Ingest: declared header length, time reconstruction, scoped nulls."""

from __future__ import annotations

import numpy as np
import pytest

from polyrmc.state import TimeFormat
from polyrmc.tier0.argen_io import (
    detect_delimiter,
    detect_time_format,
    find_declared_header_length,
    identify_scattering_columns,
    load_argen_file,
    reconstruct_time,
    scoped_null_mask,
)

from conftest import write_argen_file


def test_header_length_is_read_from_the_file(argen_file):
    lines = argen_file.read_text(encoding="utf-8").splitlines()
    assert find_declared_header_length(lines) == 5


def test_declared_count_includes_the_column_name_row(argen_file):
    """"Header Rows: N" puts the column names on line N, with data from N+1.

    Reading it as N-rows-then-header silently promotes the first data row to
    be the header, which loses a sample and turns every column into strings.
    """
    argen = load_argen_file(argen_file)

    assert "Elapsed Time" in argen.data.columns
    assert argen.data["ls8"].dtype.kind == "f"


def test_delimiter_is_detected_from_the_column_row():
    assert detect_delimiter("a,b,c,d") == ","
    assert detect_delimiter("a\tb\tc") == "\t"
    with pytest.raises(ValueError, match="no recognised delimiter"):
        detect_delimiter("nodelimitershere")


def test_ls_channels_are_recognised():
    """The instrument names its scattering channels ls1..ls16."""
    columns = ["sampleid", "elapsedtime", "stirringspeed", "temperature", "ndfilter"]
    columns += [f"ls{i}" for i in range(1, 17)]

    found = identify_scattering_columns(columns)

    assert found == [f"ls{i}" for i in range(1, 17)]
    assert "sampleid" not in found


def test_missing_header_declaration_is_an_error_not_a_guess(tmp_path):
    path = tmp_path / "undeclared.txt"
    path.write_text("Elapsed Time\tCell_1\n00:01.0\t100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not declare its header length"):
        load_argen_file(path)


@pytest.mark.parametrize(
    "tokens, expected",
    [
        (["00:01.0", "00:02.0"], TimeFormat.WRAPPING_MMSS),
        (["01:00:01", "01:00:02"], TimeFormat.ABSOLUTE_HHMMSS),
        (["1.0", "2.0"], TimeFormat.SECONDS),
    ],
)
def test_time_format_detected_from_token_shape(tokens, expected):
    assert detect_time_format(tokens) is expected


def test_wrapping_time_is_dewrapped_across_the_hour_boundary():
    tokens = ["59:58.0", "59:59.0", "00:00.0", "00:01.0"]
    time_s = reconstruct_time(tokens, TimeFormat.WRAPPING_MMSS)

    assert np.all(np.diff(time_s) > 0), "de-wrapped time must be monotonic"
    np.testing.assert_allclose(np.diff(time_s), [1.0, 1.0, 1.0])


def test_absolute_time_is_parsed_without_dewrapping():
    tokens = ["00:59:59", "01:00:00", "01:00:01"]
    time_s = reconstruct_time(tokens, TimeFormat.ABSOLUTE_HHMMSS)
    np.testing.assert_allclose(time_s, [3599.0, 3600.0, 3601.0])


def test_nulls_drop_rows_only_when_they_are_in_a_scattering_channel():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "Temperature": [25.0, np.nan, 25.0],
            "ls8": [100.0, 101.0, np.nan],
        }
    )
    mask = scoped_null_mask(frame, ["ls8"])

    assert not mask[1], "a null temperature must never discard a row"
    assert mask[2], "a null in a scattering channel must drop the row"


def test_load_applies_scoped_nulls_and_reconstructs_time(argen_file):
    argen = load_argen_file(argen_file)

    assert argen.n_header_rows == 5
    assert len(argen.available_channels) == 16
    assert argen.scattering_columns == ["ls8"], "only ls8 is analysed"
    assert argen.time_format is TimeFormat.WRAPPING_MMSS
    assert argen.dropped_rows == [10, 11], "only the channel nulls should drop rows"
    assert argen.n_points == 4000 - 2
    assert np.all(np.diff(argen.time_s) > 0)
    assert argen.duration_s > 3600, "fixture must cross an hour boundary to be useful"


def test_interior_drops_make_sampling_non_uniform(argen_file):
    """Row index and elapsed time diverge -- which is why time is the x-axis."""
    argen = load_argen_file(argen_file)
    assert not argen.is_uniformly_sampled

    gap = np.diff(argen.time_s)
    assert gap.max() > gap.min(), "the dropped rows should leave a wider interval"


def test_metadata_is_parsed_from_the_preamble(argen_file):
    argen = load_argen_file(argen_file)
    assert "ARGEN" in argen.metadata.get("Instrument", "")


def test_fewer_than_sixteen_channels_still_loads(tmp_path):
    path = write_argen_file(tmp_path / "eight_cells.txt", n_points=200, n_channels=8)
    argen = load_argen_file(path)
    assert len(argen.available_channels) == 8
    assert argen.scattering_columns == ["ls8"]


def test_only_ls8_is_kept_and_the_rest_are_dropped(argen_file):
    """The lab analyses ls8 and nothing else, so nothing else survives load."""
    argen = load_argen_file(argen_file)

    assert argen.scattering_columns == ["ls8"]
    assert "ls8" in argen.data.columns
    for other in argen.available_channels:
        if other != "ls8":
            assert other not in argen.data.columns, f"{other} must not reach analysis"
    # Auxiliary columns are untouched.
    assert "Temperature" in argen.data.columns


def test_a_null_in_an_unused_channel_does_not_drop_a_row(tmp_path):
    """A bad ls3 reading must not discard a row whose ls8 value is fine."""
    path = write_argen_file(
        tmp_path / "null_in_ls3.txt",
        n_points=200,
        null_channel_index=2,  # ls3, which is not analysed
        null_channel_rows=(30, 31, 32),
    )
    argen = load_argen_file(path)

    assert argen.dropped_rows == []
    assert argen.n_points == 200


def test_an_unknown_channel_names_what_is_available(argen_file):
    with pytest.raises(ValueError, match="is not a scattering channel"):
        load_argen_file(argen_file, channel="ls99")
