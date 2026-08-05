"""Section 13 check 1: identical inputs produce a byte-identical output CSV.

This is the check that catches nondeterminism entering the pipeline -- a stray
dict ordering, an unseeded random draw, a set iteration. It cannot live inside
``verify_output``, which sees one run; it needs two.
"""

from __future__ import annotations

import pytest

from polyrmc.pipeline import run_part1, run_part2
from polyrmc.tier1.judge import StaticJudge


def _body(path) -> str:
    """The data table alone, with the header stripped.

    The header carries a run id that is expected to differ between invocations;
    the numbers are what must not.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    declared = int(lines[0].split(":")[1])
    return "\n".join(lines[declared - 1 :])


def test_the_same_input_gives_a_byte_identical_table(run_config, tmp_path):
    first = run_part1(run_config, judge=StaticJudge(), output_csv=tmp_path / "a.csv")[1]
    second = run_part1(run_config, judge=StaticJudge(), output_csv=tmp_path / "b.csv")[1]

    assert _body(first) == _body(second)


def test_five_runs_agree(run_config, tmp_path):
    """Determinism is one of the two numbers the regression suite reports."""
    bodies = {
        _body(
            run_part1(
                run_config, judge=StaticJudge(), output_csv=tmp_path / f"r{i}.csv"
            )[1]
        )
        for i in range(5)
    }
    assert len(bodies) == 1


def test_the_header_differs_only_where_it_should(run_config, tmp_path):
    first = run_part1(run_config, judge=StaticJudge(), output_csv=tmp_path / "a.csv")[1]
    second = run_part1(run_config, judge=StaticJudge(), output_csv=tmp_path / "b.csv")[1]

    def header(path):
        lines = path.read_text(encoding="utf-8").splitlines()
        return lines[: int(lines[0].split(":")[1]) - 1]

    assert header(first) == header(second)


def test_part2_parameters_are_reproducible(run_config, virial_argen_file, tmp_path):
    config = run_config.model_copy(update={"source_file": virial_argen_file})

    first_csv = run_part1(config, judge=StaticJudge(), output_csv=tmp_path / "v1.csv")[1]
    second_csv = run_part1(config, judge=StaticJudge(), output_csv=tmp_path / "v2.csv")[1]

    first = run_part2(first_csv, config, judge=StaticJudge())
    second = run_part2(second_csv, config, judge=StaticJudge())

    assert first.parameters.a2_mw == pytest.approx(second.parameters.a2_mw, rel=0, abs=0)
    assert first.fit.n_points == second.fit.n_points


def test_state_paths_are_reproducible(run_config, tmp_path):
    """The branch taken at each open question must not vary between runs."""
    first = run_part1(run_config, judge=StaticJudge(), output_csv=tmp_path / "a.csv")[0]
    second = run_part1(run_config, judge=StaticJudge(), output_csv=tmp_path / "b.csv")[0]

    assert [(p.question, p.decision) for p in first.paths] == [
        (p.question, p.decision) for p in second.paths
    ]
