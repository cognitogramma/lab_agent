"""Reading KaleidaGraph .QDA files.

The fixtures here are written by :func:`build_qda`, which lays out the format
exactly as :mod:`continuum_spectranalytics.kaleidagraph` documents it. That keeps the suite off
the lab's real files, and it means the writer and the reader have to agree about
a format neither of them owns -- if the description in the module docstring is
wrong, these tests are wrong in the same way and the disagreement shows up the
next time a real file is opened.

Every case below is one that a real file in this lab actually exhibits.
"""

from __future__ import annotations

import numpy as np
import pytest

from continuum_spectranalytics.kaleidagraph import (
    COLUMN_RECORD_SIGNATURE,
    COLUMN_RECORD_SIZE,
    NotAQdaFile,
    NotAQpcFile,
    read_qda,
    read_qpc,
)
from continuum_spectranalytics.tabular import UnsupportedFile, open_table


def build_qda(
    columns: dict[str, list[float]],
    *,
    declared_columns: int | None = None,
    blanks: dict[str, list[int]] | None = None,
    filler: bytes = b"",
) -> bytes:
    """Write a .QDA holding the given columns.

    ``filler`` is inserted before the trailing empty columns, standing in for
    the stored plot layout a real file carries there.
    """
    blanks = blanks or {}
    n_columns = declared_columns or len(columns)
    n_rows = max((len(v) for v in columns.values()), default=0)

    name_table_at = 0x200 + n_columns * 6
    data_at = name_table_at + n_columns * 40

    head = bytearray(data_at)
    head[2:4] = n_columns.to_bytes(2, "big")
    head[0x200:0x204] = n_rows.to_bytes(4, "big")
    for index, name in enumerate(columns):
        if index >= n_columns:
            break
        start = name_table_at + index * 40
        head[start : start + len(name)] = name.encode("mac-roman")

    def record(name: str) -> bytes:
        block = bytearray(COLUMN_RECORD_SIZE)
        block[0:8] = COLUMN_RECORD_SIGNATURE
        block[8 : 8 + len(name)] = name.encode("mac-roman")
        return bytes(block)

    body = bytearray()
    for name, values in columns.items():
        body += np.asarray(values, dtype=">f4").tobytes()
        flags = np.ones(len(values), dtype=">u2")
        for row in blanks.get(name, []):
            flags[row] = 0
        body += flags.tobytes()
        body += record(name)

    body += filler
    for index in range(len(columns), n_columns):
        body += record("")

    return bytes(head) + bytes(body)


@pytest.fixture
def two_columns(tmp_path):
    path = tmp_path / "run.QDA"
    path.write_bytes(
        build_qda({"time (sec)": [0.0, 5.0, 10.0], "conductivity": [1.5, 2.5, 3.5]})
    )
    return path


def test_names_and_values_come_back(two_columns):
    parsed = read_qda(two_columns)

    assert list(parsed.frame.columns) == ["time (sec)", "conductivity"]
    assert parsed.frame["time (sec)"].tolist() == [0.0, 5.0, 10.0]
    assert parsed.frame["conductivity"].tolist() == [1.5, 2.5, 3.5]


def test_a_blank_cell_is_nan_not_the_number_in_the_slot(tmp_path):
    """A real file has one of these. The float in a blank slot is not data."""
    path = tmp_path / "gap.QDA"
    path.write_bytes(
        build_qda({"x": [1.0, 2.0, 3.0], "y": [9.0, 8.0, 7.0]}, blanks={"y": [1]})
    )

    parsed = read_qda(path)

    assert parsed.frame["y"].tolist()[0] == 9.0
    assert np.isnan(parsed.frame["y"].tolist()[1])
    assert any("blank in KaleidaGraph" in note for note in parsed.notes)


def test_columns_of_different_length_are_padded_not_truncated(tmp_path):
    path = tmp_path / "ragged.QDA"
    path.write_bytes(build_qda({"long": [1.0, 2.0, 3.0], "short": [7.0, 8.0]}))

    parsed = read_qda(path)

    assert len(parsed.frame) == 3
    assert parsed.frame["long"].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(parsed.frame["short"].tolist()[2])
    assert any("different lengths" in note for note in parsed.notes)


def test_a_wide_file_reads_its_names_from_the_right_place(tmp_path):
    """The name table moves with the column count; a fixed offset shifts it.

    A 21-column file puts its names 66 bytes past where a 10-column file does.
    Reading the old offset lands mid-table, decodes as tidy small floats, and
    produces a table that looks entirely fine.
    """
    columns = {"sampleid": [1.0, 2.0], "temperature": [22.5, 22.6]}
    columns.update({f"ls{i}": [0.01 * i, 0.02 * i] for i in range(1, 17)})
    path = tmp_path / "wide.QDA"
    path.write_bytes(build_qda(columns, declared_columns=21))

    parsed = read_qda(path)

    assert parsed.n_columns_declared == 21
    assert list(parsed.frame.columns)[:2] == ["sampleid", "temperature"]
    assert "ls16" in parsed.frame.columns
    # Compared loosely on purpose: 22.6 comes back as 22.600000381469727,
    # because the file stores single precision. Anything downstream that
    # compares these exactly is asserting against KaleidaGraph's rounding.
    assert parsed.frame["temperature"].tolist() == pytest.approx([22.5, 22.6], rel=1e-6)


def test_a_stored_plot_is_not_read_as_a_column(tmp_path):
    """Real files keep plot layout between the data and the empty columns."""
    path = tmp_path / "plotted.QDA"
    plot = bytes(range(256)) * 40  # not a flag mask, whatever its length
    path.write_bytes(
        build_qda({"x": [1.0, 2.0, 3.0]}, declared_columns=3, filler=plot)
    )

    parsed = read_qda(path)

    assert parsed.frame["x"].tolist() == [1.0, 2.0, 3.0]
    # The filler must not have become a column of plausible numbers.
    for name in parsed.frame.columns:
        if name != "x":
            assert parsed.frame[name].isna().all()


def test_the_record_name_wins_over_the_table_position(tmp_path):
    """A column with no record shifts every name after it by one.

    That is what an ARGEN export does: its text ``elapsedtime`` column is stored
    differently and leaves no record, so naming by table position labels every
    following column with its neighbour's name -- and a temperature column
    called "stirring speed" plots perfectly well.
    """
    raw = bytearray(build_qda({"first": [1.0, 2.0], "second": [3.0, 4.0]}))
    # Rename the table's first slot without touching either record.
    name_table_at = 0x200 + 2 * 6
    raw[name_table_at : name_table_at + 40] = b"wrong".ljust(40, b"\x00")
    path = tmp_path / "shifted.QDA"
    path.write_bytes(bytes(raw))

    parsed = read_qda(path)

    assert "wrong" not in parsed.frame.columns
    assert parsed.frame["first"].tolist() == [1.0, 2.0]


def test_an_empty_window_reads_as_no_data(tmp_path):
    path = tmp_path / "empty.QDA"
    path.write_bytes(build_qda({}, declared_columns=10))

    parsed = read_qda(path)

    assert parsed.n_rows_declared == 0
    assert parsed.frame.empty


def test_something_that_is_not_a_qda_is_refused(tmp_path):
    path = tmp_path / "nope.QDA"
    path.write_bytes(b"\x00" * 4096)

    with pytest.raises(NotAQdaFile, match="no KaleidaGraph column records"):
        read_qda(path)


# --- through the file layer -------------------------------------------------


def test_open_table_recognises_a_qda(two_columns):
    table = open_table(two_columns)

    assert table.kind == "kaleidagraph"
    assert table.numeric_columns() == ["time (sec)", "conductivity"]


# The .QDR case is covered below, under its real answer: there is no such
# KaleidaGraph format. This file used to assert that it was a "plot layout",
# which is what everyone assumes and what the application's own file dialogs
# contradict.


# --- .QPC, the plot file ----------------------------------------------------
#
# Everything below turns on the two ways this format stores text. Labels are
# Pascal strings and fit equations are C strings, which is why one scan cannot
# find both, and why a short label written over a longer one leaves a tail that
# a naive reader glues on.


def pascal(text: str) -> bytes:
    return bytes([len(text)]) + text.encode("mac-roman")


def build_qpc(
    *,
    labels: list[str] | None = None,
    fit: str | None = None,
    source: str | None = None,
    fonts: list[str] | None = None,
    stale_tail: str = "",
    header: bytes = b"\x00\x0f",
) -> bytes:
    """Write a .QPC holding the given plot description.

    ``stale_tail`` is written straight after the first label without a
    terminator, reproducing what KaleidaGraph leaves behind when a short value
    overwrites a longer one.
    """
    out = bytearray(header + b"\x00\x00" + bytes(508))
    for name in fonts or []:
        out += pascal(name) + bytes(24)
    for index, text in enumerate(labels or []):
        out += pascal(text)
        if index == 0 and stale_tail:
            out += stale_tail.encode("mac-roman")
        out += bytes(64)
    if source:
        out += pascal(source) + bytes(64)
    if fit:
        # A C string, as the application stores equations.
        out += b"\x00" + fit.encode("mac-roman") + b"\x00" + bytes(64)
    return bytes(out)


def test_a_plot_reports_its_titles_and_fit(tmp_path):
    path = tmp_path / "graph.QPC"
    path.write_bytes(
        build_qpc(
            labels=["Displacement (mm)", "Remote Applied Stress (MPa)"],
            fit="m1 + m2*exp((m0-m3)*(m0-m3)/(m4*m4)); m1=1; m2=1; m3=1; m4=1",
            source="91310-1.norm 9:30:25 AM 3/16/93",
            fonts=["Helvetica"],
        )
    )

    plot = read_qpc(path)

    assert "Displacement (mm)" in plot.labels
    assert "Remote Applied Stress (MPa)" in plot.labels
    assert plot.fit_equations[0].endswith("m4=1")
    assert plot.source_references == ["91310-1.norm 9:30:25 AM 3/16/93"]
    assert "Helvetica" not in plot.labels  # fonts are not what anyone asked for


def test_a_stale_tail_is_not_glued_onto_the_label(tmp_path):
    """The exact bug a real file shows: 'Displacement (mm)ress (MPa)'."""
    path = tmp_path / "overwritten.QPC"
    path.write_bytes(
        build_qpc(labels=["Displacement (mm)"], stale_tail="ress (MPa)")
    )

    plot = read_qpc(path)

    assert "Displacement (mm)" in plot.labels
    assert not any("mm)ress" in text for text in plot.labels + plot.other_text)


def test_the_older_header_version_is_still_a_plot(tmp_path):
    """Two of the fifteen shipped examples open with 000c, same as a data file.

    Which is why the header cannot be the thing that tells them apart.
    """
    path = tmp_path / "old.QPC"
    path.write_bytes(build_qpc(labels=["Wavelength, nm"], header=b"\x00\x0c"))

    assert "Wavelength, nm" in read_qpc(path).labels


def test_a_data_file_is_not_read_as_a_plot(tmp_path):
    path = tmp_path / "actually_data.QPC"
    path.write_bytes(build_qda({"x": [1.0, 2.0]}))

    with pytest.raises(NotAQpcFile, match="column records"):
        read_qpc(path)


def test_a_fit_fragment_never_outranks_the_whole_equation(tmp_path):
    """A non-printable byte inside an equation leaves a valid-looking tail."""
    whole = "m1 + m2*exp((m0-m3)*(m0-m3)/(m4*m4)); m1=1; m2=1; m3=1; m4=1"
    raw = bytearray(build_qpc(labels=["Signal"], fit=whole))
    raw += b"\x00" + b"m4*m4)); m1=1; m2=1" + b"\x00"
    path = tmp_path / "split.QPC"
    path.write_bytes(bytes(raw))

    plot = read_qpc(path)

    assert plot.fit_equations[0] == whole


def test_a_plot_is_refused_as_a_table_and_says_where_the_data_is(tmp_path):
    path = tmp_path / "graph.qpc"
    path.write_bytes(build_qpc(labels=["Signal"]))

    with pytest.raises(UnsupportedFile) as error:
        open_table(path)

    message = str(error.value)
    assert "no rows in it" in message
    assert ".QDA" in message


def test_qdr_is_answered_precisely_because_it_does_not_exist(tmp_path):
    """It is asked for by name often enough to deserve a real answer."""
    path = tmp_path / "figure.qdr"
    path.write_bytes(b"\x00binary")

    with pytest.raises(UnsupportedFile) as error:
        open_table(path)

    message = str(error.value)
    assert "not a KaleidaGraph format" in message
    assert ".QPC" in message and ".QDA" in message
