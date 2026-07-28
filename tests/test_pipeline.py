"""End-to-end runs, provenance, and the orchestrator's confirmation gate."""

from __future__ import annotations

import numpy as np
import pytest

from polyrmc.config import FitRangeConfig, LoopConfig, RunConfig, SmoothingConfig
from polyrmc.io_csv import read_processed_csv
from polyrmc.pipeline import new_run_id, run_part1, run_part2
from polyrmc.provenance import load_sidecar, replay_decisions, sidecar_path
from polyrmc.tier1.judge import RecordedJudge, StaticJudge
from polyrmc.tier2.orchestrator import Orchestrator, PendingRun

from conftest import TRUE_A2, TRUE_MW

TRUE_A2_MW = TRUE_A2 * TRUE_MW


@pytest.fixture
def virial_config(virial_argen_file, optical, dilution):
    return RunConfig(
        run_id="virial-test",
        source_file=virial_argen_file,
        experiment_type="dilution_trajectory",
        optical=optical,
        dilution=dilution,
        smoothing=SmoothingConfig(min_window=5, max_window=51, n_candidates=5),
        fit_range=FitRangeConfig(min_points=50, n_candidates=5),
        loop=LoopConfig(max_iterations=2),
    )


def test_part1_emits_a_csv_and_a_sidecar(run_config, tmp_path):
    output = tmp_path / "out.csv"
    state, path = run_part1(run_config, judge=StaticJudge(), output_csv=output)

    assert path.exists()
    assert sidecar_path(path).exists()
    assert state.smoothed_signal is not None
    assert state.smoothed_signal.size == state.raw_signal.size


def test_part1_preserves_the_time_axis_length(run_config, tmp_path):
    state, _path = run_part1(
        run_config, judge=StaticJudge(), output_csv=tmp_path / "out.csv"
    )

    assert state.time_s.size == state.raw_signal.size
    assert state.time_s.size == state.corrected_signal.size
    assert np.all(np.diff(state.time_s) > 0)


def test_sidecar_records_the_loop_trajectory(run_config, tmp_path):
    output = tmp_path / "out.csv"
    run_part1(run_config, judge=StaticJudge(), output_csv=output)

    record = load_sidecar(sidecar_path(output))

    assert record["run_id"] == run_config.run_id
    assert record["ingest"]["time_format"] == "wrapping_mmss"
    assert record["ingest"]["dropped_rows"] == [10, 11]
    assert "smoothing_window" in record["loops"]

    loop = record["loops"]["smoothing_window"]
    assert loop["resolved_value"] is not None
    assert loop["decisions"], "decisions must be recorded for replay"
    assert loop["candidates_by_iteration"], "the offered set must be recorded too"


def test_recorded_decisions_replay_to_the_same_window(run_config, tmp_path):
    output = tmp_path / "out.csv"
    first, _path = run_part1(run_config, judge=StaticJudge(), output_csv=output)

    record = load_sidecar(sidecar_path(output))
    decisions = replay_decisions(record, "smoothing_window")

    second, _path = run_part1(
        run_config, judge=RecordedJudge(decisions), output_csv=tmp_path / "replay.csv"
    )

    assert second.loops["smoothing_window"].resolved_value == (
        first.loops["smoothing_window"].resolved_value
    )
    assert second.loops["smoothing_window"].provenance[0].replayed is True


def test_uncalibrated_run_warns_and_withholds_absolute_values(virial_config, tmp_path):
    _state, csv_path = run_part1(
        virial_config, judge=StaticJudge(), output_csv=tmp_path / "v.csv"
    )
    state = run_part2(csv_path, virial_config, judge=StaticJudge())

    assert state.parameters.calibrated is False
    assert state.parameters.mw is None
    assert state.parameters.a2 is None
    assert any("alpha" in warning for warning in state.warnings)


def test_end_to_end_recovers_the_known_a2_mw(virial_config, tmp_path):
    """The whole point: a known answer survives the full pipeline uncalibrated."""
    _state, csv_path = run_part1(
        virial_config, judge=StaticJudge(), output_csv=tmp_path / "v.csv"
    )
    state = run_part2(csv_path, virial_config, judge=StaticJudge())

    assert state.parameters.a2_mw == pytest.approx(TRUE_A2_MW, rel=0.10)
    assert state.fit.n_points >= virial_config.fit_range.min_points


def test_alpha_calibration_does_not_move_a2_mw(virial_config, tmp_path):
    """Setting alpha must change Mw and A2 but leave their product alone."""
    _state, uncalibrated_csv = run_part1(
        virial_config, judge=StaticJudge(), output_csv=tmp_path / "uncal.csv"
    )
    uncalibrated = run_part2(uncalibrated_csv, virial_config, judge=StaticJudge())

    calibrated_config = virial_config.model_copy(deep=True)
    calibrated_config.optical.alpha = 1.0e9
    _state, calibrated_csv = run_part1(
        calibrated_config, judge=StaticJudge(), output_csv=tmp_path / "cal.csv"
    )
    calibrated = run_part2(calibrated_csv, calibrated_config, judge=StaticJudge())

    assert calibrated.parameters.calibrated is True
    assert calibrated.parameters.a2_mw == pytest.approx(
        uncalibrated.parameters.a2_mw, rel=1e-6
    )
    assert calibrated.parameters.mw == pytest.approx(TRUE_MW, rel=0.10)


def test_part2_refuses_a_file_with_no_concentration_axis(run_config, tmp_path):
    fixed = run_config.model_copy(update={"experiment_type": "fixed_concentration"})
    _state, csv_path = run_part1(fixed, judge=StaticJudge(), output_csv=tmp_path / "f.csv")

    with pytest.raises(ValueError, match="requires a dilution_trajectory run"):
        run_part2(csv_path, fixed, judge=StaticJudge())


def test_unknown_channel_is_refused(run_config, tmp_path):
    with pytest.raises(ValueError, match="is not a scattering channel"):
        run_part1(run_config, channel="ls99", judge=StaticJudge())


def test_ls8_is_the_default_channel(run_config, tmp_path):
    """No channel argument means ls8; nothing else is ever analysed."""
    assert run_config.channel == "ls8"

    _state, csv_path = run_part1(
        run_config, judge=StaticJudge(), output_csv=tmp_path / "default.csv"
    )
    _frame, metadata = read_processed_csv(csv_path)

    assert metadata["channel"] == "ls8"


def test_orchestrator_refuses_to_run_without_a_confirmed_experiment_type(run_config):
    orchestrator = Orchestrator(template=run_config, judge=StaticJudge())
    pending = PendingRun(path=run_config.source_file)

    with pytest.raises(RuntimeError, match="no confirmed experiment type"):
        orchestrator.execute(pending)


def test_orchestrator_runs_once_a_human_confirms(run_config, tmp_path):
    orchestrator = Orchestrator(template=run_config, judge=StaticJudge())
    orchestrator.pending.append(PendingRun(path=run_config.source_file))

    orchestrator.confirm_all(lambda run: "fixed_concentration")
    state = orchestrator.execute(orchestrator.pending[0])

    assert state in orchestrator.completed
    assert "run" in orchestrator.report(state)


def test_orchestrator_discovery_stages_without_executing(run_config, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "a.txt").write_text("placeholder", encoding="utf-8")
    (inbox / "b.txt").write_text("placeholder", encoding="utf-8")
    orchestrator = Orchestrator(template=run_config)

    staged = orchestrator.discover(inbox)

    assert len(staged) == 2
    assert all(not run.is_confirmed for run in staged)
    assert orchestrator.completed == []


def test_new_run_id_is_unique():
    assert new_run_id() != new_run_id()
