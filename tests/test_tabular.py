"""Opening a data file, and plotting columns out of it.

This is the path a researcher takes before anything has been declared or
processed: here is a file, what is in it, show me. Almost everything worth
testing is a message -- what the tool says when the file is a format it cannot
read, or when the column asked for is the text one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from polyrmc import figures
from polyrmc.cli import main
from polyrmc.tabular import UnsupportedFile, open_table


@pytest.fixture
def delimited(tmp_path) -> Path:
    path = tmp_path / "export.txt"
    path.write_text("time\tsignal\tlabel\n0\t1.5\ta\n1\t2.5\tb\n", encoding="utf-8")
    return path


def test_a_raw_acquisition_opens_with_every_channel(argen_file):
    """The pipeline narrows to ls8. Looking at a file is when the rest matter."""
    table = open_table(argen_file)

    assert table.kind == "argen"
    assert "seconds" in table.columns
    assert "ls1" in table.columns and "ls16" in table.columns
    assert any("nothing is spliced" in n for n in table.notes)


def test_the_reconstructed_time_is_the_plottable_one(argen_file):
    """The instrument's own time column is text that wraps every hour."""
    table = open_table(argen_file)

    assert "seconds" in table.numeric_columns()
    assert "Elapsed Time" not in table.numeric_columns()


def test_a_processed_csv_is_recognised_as_one(tmp_path, run_config):
    from tests.test_rehydrate import write_run

    write_run(tmp_path / "runs", "done_ls8", "done")

    table = open_table(tmp_path / "runs" / "done_ls8.csv")

    assert table.kind == "processed"
    assert "Mw_over_M0" in table.columns


def test_an_ordinary_export_is_sniffed_and_says_so(delimited):
    table = open_table(delimited)

    assert table.kind == "delimited"
    assert table.columns == ["time", "signal", "label"]
    assert table.numeric_columns() == ["time", "signal"]
    assert any("not recognised" in n for n in table.notes)


def test_unreadable_formats_are_refused_with_what_to_do(tmp_path):
    """A refusal that does not say what to try instead is just a dead end."""
    path = tmp_path / "old_experiment.xlsx"
    path.write_bytes(b"\x00\x01binary")

    with pytest.raises(UnsupportedFile) as error:
        open_table(path)

    message = str(error.value)
    assert "Excel" in message
    assert "Export it to text" in message  # actionable, not just a refusal


def test_a_missing_file_says_so_plainly(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_table(tmp_path / "nothing.txt")


def test_a_directory_is_not_a_data_file(tmp_path):
    with pytest.raises(IsADirectoryError):
        open_table(tmp_path)


def test_a_file_with_no_columns_is_refused(tmp_path):
    path = tmp_path / "prose.txt"
    path.write_text("this is not a table\nit is a sentence\n", encoding="utf-8")

    with pytest.raises(ValueError, match="could not be read as a table"):
        open_table(path)


# --- the exploratory plot ---------------------------------------------------


def test_columns_plot_writes_a_png(tmp_path, argen_file):
    table = open_table(argen_file)

    out = figures.plot_columns(
        table.frame, "seconds", ["ls8", "ls1"], tmp_path / "cols.png",
        source="argen_file (argen)",
    )

    assert out.exists() and out.stat().st_size > 5_000


def test_an_unknown_column_lists_what_is_available(tmp_path, argen_file):
    table = open_table(argen_file)

    with pytest.raises(ValueError) as error:
        figures.plot_columns(table.frame, "seconds", ["ls99"], tmp_path / "x.png")

    message = str(error.value)
    assert "ls99" in message
    assert "seconds" in message  # the available list follows


def test_a_text_column_cannot_go_on_an_axis(tmp_path, argen_file):
    table = open_table(argen_file)

    with pytest.raises(ValueError, match="not numeric"):
        figures.plot_columns(table.frame, "Elapsed Time", ["ls8"], tmp_path / "x.png")


def test_nothing_to_plot_is_refused(tmp_path, argen_file):
    table = open_table(argen_file)

    with pytest.raises(ValueError, match="at least one y column"):
        figures.plot_columns(table.frame, "seconds", [], tmp_path / "x.png")


# --- the CLI stage, which needs no model and no key -------------------------


def test_cli_describes_a_file_when_given_no_columns(argen_file, capsys):
    code = main(["plot", "--file", str(argen_file)])

    out = capsys.readouterr().out
    assert code == 0
    assert "plottable columns:" in out
    assert "ls8" in out
    assert "polyrmc plot --file" in out  # tells you the next command


def test_cli_writes_a_figure_beside_the_file(argen_file, capsys):
    code = main(["plot", "--file", str(argen_file), "--x", "seconds", "--y", "ls8"])

    assert code == 0
    assert (argen_file.parent / f"{argen_file.stem}_seconds_vs_ls8.png").exists()
    assert "wrote" in capsys.readouterr().out


def test_cli_honours_an_explicit_output_path(argen_file, tmp_path):
    target = tmp_path / "named.png"

    main(["plot", "--file", str(argen_file), "--x", "seconds", "--y", "ls8",
          "--out", str(target), "--style", "scatter", "--logy"])

    assert target.exists()


def test_cli_reports_an_unreadable_format_without_a_traceback(tmp_path, capsys):
    path = tmp_path / "legacy.qdr"
    path.write_bytes(b"\x00binary")

    code = main(["plot", "--file", str(path)])

    assert code == 2
    assert "KaleidaGraph" in capsys.readouterr().out


def test_cli_needs_a_file(capsys):
    assert main(["plot"]) == 2
    assert "needs --file" in capsys.readouterr().out
