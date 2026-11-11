"""End-to-end runs and the provenance they leave behind."""

from __future__ import annotations

import numpy as np
import pytest

from continuum_spectranalytics.config import FitRangeConfig, LoopConfig, RunConfig, SmoothingConfig
from continuum_spectranalytics.io_csv import read_processed_csv
from continuum_spectranalytics.pipeline import new_run_id, run_part1, run_part2
from continuum_spectranalytics.provenance import load_sidecar, sidecar_path

from conftest import TRUE_A2, TRUE_MW

TRUE_A2_MW = TRUE_A2 * TRUE_MW


@pytest.fixture
def virial_config(virial_argen_file, optical, dilution):
    return RunConfig(
        run_id="virial-test",
        source_file=virial_argen_file,
        experiment_type="acd_dilution",
        optical=optical,
        dilution=dilution,
        smoothing=SmoothingConfig(min_window=5, max_window=51, n_candidates=5),
        fit_range=FitRangeConfig(min_points=50, n_candidates=5),
        loop=LoopConfig(max_iterations=2))


def test_part1_emits_a_csv_and_a_sidecar(run_config, tmp_path):
    output = tmp_path / "out.csv"
    state, path = run_part1(run_config, output_csv=output)

    assert path.exists()
    assert sidecar_path(path).exists()
    assert state.smoothed_signal is not None
    assert state.smoothed_signal.size == state.raw_signal.size


def test_part1_preserves_the_time_axis_length(run_config, tmp_path):
    state, _path = run_part1(
        run_config, output_csv=tmp_path / "out.csv"
    )

    assert state.time_s.size == state.raw_signal.size
    assert state.time_s.size == state.corrected_signal.size
    assert np.all(np.diff(state.time_s) > 0)


def test_sidecar_records_the_loop_trajectory(run_config, tmp_path):
    output = tmp_path / "out.csv"
    run_part1(run_config, output_csv=output)

    record = load_sidecar(sidecar_path(output))

    assert record["run_id"] == run_config.run_id
    assert record["ingest"]["time_format"] == "wrapping_mmss"
    assert record["ingest"]["dropped_rows"] == [10, 11]
    assert "smoothing_window" in record["loops"]

    loop = record["loops"]["smoothing_window"]
    assert loop["resolved_value"] is not None
    # The offered set is the part that has to survive. Which windows passed the
    # gates is a claim about the data, and a reviewer checks the chosen one
    # against the others -- so recording only the winner would be recording the
    # conclusion without the evidence.
    assert loop["candidates_by_iteration"], "the offered set must be recorded"
    offered = loop["candidates_by_iteration"][0]
    assert any(
        candidate["value"] == loop["resolved_value"] for candidate in offered
    ), "the chosen window must be one of the candidates that was certified"


def test_uncalibrated_run_warns_and_withholds_absolute_values(virial_config, tmp_path):
    _state, csv_path = run_part1(
        virial_config, output_csv=tmp_path / "v.csv"
    )
    state = run_part2(csv_path, virial_config)

    assert state.parameters.calibrated is False
    assert state.parameters.mw is None
    assert state.parameters.a2 is None
    assert any("alpha" in warning for warning in state.warnings)


def test_end_to_end_recovers_the_known_a2_mw(virial_config, tmp_path):
    """The whole point: a known answer survives the full pipeline uncalibrated."""
    _state, csv_path = run_part1(
        virial_config, output_csv=tmp_path / "v.csv"
    )
    state = run_part2(csv_path, virial_config)

    assert state.parameters.a2_mw == pytest.approx(TRUE_A2_MW, rel=0.10)
    assert state.fit.n_points >= virial_config.fit_range.min_points


def test_alpha_calibration_does_not_move_a2_mw(virial_config, tmp_path):
    """Setting alpha must change Mw and A2 but leave their product alone."""
    _state, uncalibrated_csv = run_part1(
        virial_config, output_csv=tmp_path / "uncal.csv"
    )
    uncalibrated = run_part2(uncalibrated_csv, virial_config)

    calibrated_config = virial_config.model_copy(deep=True)
    # Calibrating means declaring both that the channel is a count rate and what
    # the conversion is. Either alone leaves I_rayleigh blocked.
    calibrated_config.ls_units = "count_rate"
    calibrated_config.optical.alpha = 1.0e9
    _state, calibrated_csv = run_part1(
        calibrated_config, output_csv=tmp_path / "cal.csv"
    )
    calibrated = run_part2(calibrated_csv, calibrated_config)

    assert calibrated.parameters.calibrated is True
    assert calibrated.parameters.a2_mw == pytest.approx(
        uncalibrated.parameters.a2_mw, rel=1e-6
    )
    assert calibrated.parameters.mw == pytest.approx(TRUE_MW, rel=0.10)


def test_part2_refuses_a_single_concentration(run_config, tmp_path):
    """A fixed-concentration run carries a constant c, which is one point on the axis.

    It is not a missing column -- the design puts c0 in the concentration column
    on this run type -- so the refusal has to name the real reason: one point
    has no slope, and therefore no A2.
    """
    fixed = run_config.model_copy(update={"experiment_type": "fixed_conc_aggregation"})
    _state, csv_path = run_part1(fixed, output_csv=tmp_path / "f.csv")

    with pytest.raises(ValueError, match="one concentration"):
        run_part2(csv_path, fixed)


def test_unknown_channel_is_refused(run_config, tmp_path):
    with pytest.raises(ValueError, match="is not a scattering channel"):
        run_part1(run_config, channel="ls99")


def test_ls8_is_the_default_channel(run_config, tmp_path):
    """No channel argument means ls8; nothing else is ever analysed."""
    assert run_config.channel == "ls8"

    _state, csv_path = run_part1(
        run_config, output_csv=tmp_path / "default.csv"
    )
    _frame, metadata = read_processed_csv(csv_path)

    assert metadata["channel"] == "ls8"


def test_new_run_id_is_unique():
    assert new_run_id() != new_run_id()
