"""verify_output: the checks the pipeline runs against its own work.

A check that cannot run must report ``skipped``, never ``pass``. A silently
passed check leaves a run carrying the appearance of verification it never had,
which is worse than having no check at all.
"""

from __future__ import annotations

import pytest

from polyrmc.pipeline import run_part1
from polyrmc.state import Conflict, Declaration
from polyrmc.verify import summarize, verify_output


@pytest.fixture
def verified(run_config, tmp_path):
    state, csv_path = run_part1(
        run_config, output_csv=tmp_path / "v.csv"
    )
    return state, csv_path


def by_name(results, fragment):
    return next(r for r in results if fragment in r.name)


def test_a_clean_run_has_no_failures(verified):
    state, csv_path = verified
    results = verify_output(csv_path, state)
    failures = [r for r in results if r.failed]
    assert not failures, summarize(results)


def test_every_check_reports_a_status_and_a_reason(verified):
    state, csv_path = verified
    for result in verify_output(csv_path, state):
        assert result.status in {"pass", "fail", "skipped"}
        assert result.detail, f"check {result.number} gave no reason"


def test_all_eleven_design_checks_are_present(verified):
    """Check 1 lives in the regression suite; the other ten are here, plus check 0."""
    state, csv_path = verified
    numbers = {r.number for r in verify_output(csv_path, state)}

    assert numbers == {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}


def test_monotonic_seconds_passes_on_a_real_run(verified):
    state, csv_path = verified
    assert by_name(verify_output(csv_path, state), "seconds monotonic").status == "pass"


def test_the_safe_band_check_reports_the_band(verified):
    state, csv_path = verified
    result = by_name(verify_output(csv_path, state), "safe band")
    assert result.status == "pass"
    assert "certified" in result.detail


def test_an_empty_safe_band_fails(verified):
    state, csv_path = verified
    state.loops["smoothing_window"].candidates_by_iteration = [[]]

    result = by_name(verify_output(csv_path, state), "safe band")
    assert result.failed
    assert "do not relax the tolerances" in result.detail


def test_the_residual_whitening_check_runs_on_a_real_run(verified):
    state, csv_path = verified
    result = by_name(verify_output(csv_path, state), "residual whitens")
    assert result.status == "pass"


def test_a_correlated_residual_fails(verified):
    state, csv_path = verified
    result = by_name(verify_output(csv_path, state, max_residual_acf=-1.0), "residual")
    assert result.failed
    assert "leaving real structure" in result.detail


def test_re_indexing_is_caught(verified):
    """Compacting the array shortens the time axis and distorts every derivative."""
    state, csv_path = verified
    state.time_s = state.time_s[:-10]

    result = by_name(verify_output(csv_path, state), "no re-indexing")
    assert result.failed
    assert "re-indexed" in result.detail


def test_layout_round_trip_passes(verified):
    state, csv_path = verified
    assert by_name(verify_output(csv_path, state), "layout A").status == "pass"


def test_an_unresolved_conflict_fails_verification(verified):
    state, csv_path = verified
    state.declaration_conflicts = [
        Conflict(check="experiment_type_vs_trace", declared="a", observed="b")
    ]
    result = by_name(verify_output(csv_path, state), "declarations consistent")
    assert result.failed
    assert "should not have been emitted" in result.detail


def test_a_resolved_conflict_does_not_fail(verified):
    state, csv_path = verified
    state.declarations["technique"] = Declaration(value="SLS", author="arnold")
    state.declaration_conflicts = [
        Conflict(
            check="c",
            declared="a",
            observed="b",
            resolved=True,
            resolution_reason="explained",
            resolved_by="arnold")
    ]
    assert (
        by_name(verify_output(csv_path, state), "declarations consistent").status
        == "pass"
    )


def test_checks_that_cannot_run_are_skipped_not_passed(verified):
    state, csv_path = verified
    results = verify_output(csv_path, state)

    # Nothing was censored, no drift fitted, no anchors selected on this run.
    for fragment in ("censored values", "drift slope", "anchor guard", "reversible"):
        assert by_name(results, fragment).status == "skipped"


def test_censoring_leak_is_caught(verified):
    """A value surviving at a railed index means capping or retention leaked."""
    state, csv_path = verified
    # Claim a censored index that was never masked out of the emitted columns.
    state.censored_indices = [5]

    result = by_name(verify_output(csv_path, state), "censored values")
    assert result.failed
    assert "leaked through" in result.detail


def test_a_run_with_real_censoring_passes_the_censoring_check(run_config, tmp_path):
    """The fixture opens near 4300 counts; a ceiling of 4200 clips the bright start."""
    config = run_config.model_copy(deep=True)
    config.detectors.saturation_ceiling = 4200.0
    state, csv_path = run_part1(
        config, output_csv=tmp_path / "c2.csv"
    )

    assert state.censored_indices, "expected the opening samples to be censored"
    assert by_name(verify_output(csv_path, state), "censored values").status == "pass"


def test_a_ceiling_below_the_whole_trace_is_refused(run_config, tmp_path):
    """Nothing usable survives, and the run says why rather than failing in numpy."""
    config = run_config.model_copy(deep=True)
    config.detectors.saturation_ceiling = 1.0

    with pytest.raises(ValueError, match="nothing here to analyse"):
        run_part1(config, output_csv=tmp_path / "c3.csv")


def test_m0_sensitivity_is_reported_every_run(verified):
    state, csv_path = verified
    result = by_name(verify_output(csv_path, state), "M0 sensitivity")
    assert result.status in {"pass", "skipped"}
    assert "%" in result.detail or "could not" in result.detail


def test_a_divergent_m0_must_be_flagged_to_pass(verified):
    state, csv_path = verified
    state.m0.sensitivity = 0.5
    state.m0.escalations = []
    state.m0.blocked = []

    result = by_name(verify_output(csv_path, state), "M0 sensitivity")
    assert result.failed

    state.m0.escalations = ["flagged"]
    assert by_name(verify_output(csv_path, state), "M0 sensitivity").status == "pass"


def test_summarize_renders_one_line_per_check(verified):
    state, csv_path = verified
    results = verify_output(csv_path, state)
    text = summarize(results)
    assert len(text.splitlines()) == len(results)
