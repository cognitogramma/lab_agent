"""Figures are written, and gain changes are detected rather than ignored."""

from __future__ import annotations

import numpy as np
import pytest

from polyrmc.io_csv import read_processed_csv
from polyrmc.pipeline import run_part1
from polyrmc.plotting import plot_run, plot_session
from polyrmc.tier0.argen_io import at_full_scale, gain_change_indices
from polyrmc.tier1.judge import StaticJudge


def test_gain_change_indices_finds_a_switch():
    values = np.array([0.0] * 50 + [0.5] * 50)
    assert gain_change_indices(values) == [50]


def test_a_constant_setting_reports_no_change():
    assert gain_change_indices(np.zeros(100)) == []


def test_nulls_do_not_register_as_a_gain_change():
    """The ND filter column is routinely blank; blank is not a change."""
    values = np.array([0.0, np.nan, np.nan, 0.0, 0.0])
    assert gain_change_indices(values) == []


def test_full_scale_samples_are_identified():
    values = np.array([0.3, 1.0, 0.9, 1.0])
    assert at_full_scale(values).tolist() == [False, True, False, True]


def test_plot_run_writes_a_figure(run_config, tmp_path):
    state, csv_path = run_part1(
        run_config, judge=StaticJudge(), output_csv=tmp_path / "run.csv"
    )
    frame, _metadata = read_processed_csv(csv_path)

    out = plot_run(state, run_config, frame, tmp_path / "figure.png")

    assert out.exists()
    assert out.stat().st_size > 5000, "a real figure should not be nearly empty"


def test_plot_session_marks_unreliable_runs_hollow(tmp_path):
    rows = [
        {"run_id": "a_500mM_Gd", "mw_over_m0_final": 2.0, "plateau_swing": 0.1,
         "nd_filter_changes": 0, "at_full_scale": 0},
        {"run_id": "b_3000mM_Gd", "mw_over_m0_final": 9.0, "plateau_swing": 1.6,
         "nd_filter_changes": 1, "at_full_scale": 400},
    ]
    out = plot_session(rows, tmp_path / "session.png")
    assert out.exists()


def test_plot_session_needs_at_least_one_completed_run(tmp_path):
    with pytest.raises(ValueError, match="no completed runs"):
        plot_session([{"run_id": "failed", "mw_over_m0_final": None}], tmp_path / "s.png")
