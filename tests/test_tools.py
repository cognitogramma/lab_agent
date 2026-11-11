"""The tool surface rejects invalid actions, with reasons a caller can act on."""

from __future__ import annotations

import numpy as np
import pytest

from continuum_spectranalytics.state import AnomalyRecord, AnomalyType, Candidate, Conflict, LoopState
from continuum_spectranalytics.tools import Rejected, ToolSurface
from continuum_spectranalytics.tier0.anchors import propose_anchors

RNG = np.random.default_rng(7)
NOISE = 0.002


@pytest.fixture
def surface(run_config):
    from continuum_spectranalytics.state import RunState

    time_s = np.arange(0.0, 4000.0, 1.0)
    signal = (
        0.20
        + 0.60 * np.exp(-(((time_s - 2000.0) / 250.0) ** 2))
        + 5.0e-6 * time_s
        + RNG.normal(0, NOISE, time_s.size)
    )
    state = RunState(run_id="tool-test", source_file="none")
    state.time_s = time_s
    state.raw_signal = signal
    state.spliced_signal = signal
    state.anchor_candidates, state.transition_region = propose_anchors(
        time_s, signal, noise_floor=NOISE
    )
    return ToolSurface(state=state, config=run_config)


# --- commit tier: the rejections ---------------------------------------------


def test_a_window_outside_the_safe_band_is_refused(surface):
    surface.state.loops["smoothing_window"] = LoopState(
        name="smoothing_window",
        max_iterations=3,
        candidates_by_iteration=[
            [Candidate(index=i, value=w) for i, w in enumerate([11, 15, 19])]
        ],
    )
    with pytest.raises(Rejected) as error:
        surface.pick_window(51)

    message = str(error.value)
    assert "REJECTED" in message
    # The rejection has to say what is valid instead, or the caller just guesses.
    assert "[11, 15, 19]" in message


def test_an_even_window_is_refused(surface):
    surface.state.loops["smoothing_window"] = LoopState(
        name="smoothing_window",
        max_iterations=3,
        candidates_by_iteration=[[Candidate(index=0, value=11)]],
    )
    with pytest.raises(Rejected, match="even"):
        surface.pick_window(12)


def test_a_window_from_the_band_is_accepted(surface):
    surface.state.loops["smoothing_window"] = LoopState(
        name="smoothing_window",
        max_iterations=3,
        candidates_by_iteration=[[Candidate(index=0, value=11)]],
    )
    assert surface.pick_window(11).ok
    assert surface.state.window_length == 11


def test_free_form_anchor_ids_are_refused(surface):
    with pytest.raises(Rejected) as error:
        surface.select_anchors([999], selected_by="llm")
    assert "free-form index ranges are not accepted" in str(error.value)


def test_a_single_anchor_is_refused(surface):
    first = surface.state.anchor_candidates[0].id
    with pytest.raises(Rejected, match="cannot define a trend"):
        surface.select_anchors([first], selected_by="researcher")


def test_a_valid_two_sided_selection_is_accepted(surface):
    low, high = surface.state.transition_region
    before = next(c for c in surface.state.anchor_candidates if c.index_range[1] < low)
    after = next(c for c in surface.state.anchor_candidates if c.index_range[0] > high)

    result = surface.select_anchors([before.id, after.id], selected_by="vlm")
    assert result.ok
    assert surface.state.anchor_selected_by == "vlm"


def test_correction_without_a_selection_is_refused(surface):
    with pytest.raises(Rejected, match="No validated anchor selection"):
        surface.apply_baseline_correction()


def test_a_correction_applies_and_reverts_exactly(surface):
    low, high = surface.state.transition_region
    before = next(c for c in surface.state.anchor_candidates if c.index_range[1] < low)
    after = next(c for c in surface.state.anchor_candidates if c.index_range[0] > high)
    surface.select_anchors([before.id, after.id], selected_by="researcher")

    original = surface.state.spliced_signal.copy()
    assert surface.apply_baseline_correction().ok
    assert surface.state.baseline_applied
    assert not np.allclose(surface.state.corrected_signal, original)

    assert surface.revert_baseline_correction().ok
    assert not surface.state.baseline_applied
    assert np.allclose(surface.state.spliced_signal, original, rtol=0, atol=1e-9)


def test_applying_twice_is_refused(surface):
    low, high = surface.state.transition_region
    before = next(c for c in surface.state.anchor_candidates if c.index_range[1] < low)
    after = next(c for c in surface.state.anchor_candidates if c.index_range[0] > high)
    surface.select_anchors([before.id, after.id], selected_by="researcher")
    surface.apply_baseline_correction()

    with pytest.raises(Rejected, match="already applied"):
        surface.apply_baseline_correction()


def test_correct_on_a_non_drift_region_is_refused(surface):
    surface.state.anomalies = [
        AnomalyRecord(
            start_index=10,
            end_index=12,
            channel="ls8",
            detected_by=["derivative"],
            anomaly_type=AnomalyType.TRANSIENT_SPIKE,
        )
    ]
    with pytest.raises(Rejected, match="applies only to baseline_drift"):
        surface.set_decision((10, 12), "correct")


def test_a_decision_on_an_undetected_region_is_refused(surface):
    with pytest.raises(Rejected, match="No detected region matches"):
        surface.set_decision((10, 12), "excise")


def test_an_unknown_config_key_is_refused(surface):
    with pytest.raises(Rejected, match="not a known config key"):
        surface.set_config("smoothing.polyorderr", 2)


def test_an_out_of_range_config_value_is_refused(surface):
    with pytest.raises(Rejected, match="is invalid"):
        surface.set_config("smoothing.polyorder", 9)


def test_a_valid_config_change_is_accepted(surface):
    assert surface.set_config("baseline.min_anchors", 3).ok
    assert surface.config.baseline.min_anchors == 3


def test_an_unresolved_conflict_blocks_emit(surface):
    surface.state.declaration_conflicts = [
        Conflict(check="experiment_type_vs_trace", declared="a", observed="b")
    ]
    with pytest.raises(Rejected, match="unresolved declaration conflict"):
        surface.emit_csv("out.csv")


def test_resolving_a_conflict_unblocks_the_emit_gate(surface):
    """The gate stops naming the conflict once it carries a recorded reason.

    This surface has no loaded run, so emit still refuses -- for the next reason
    down the list, not the conflict. That distinction is the point: each blocker
    is reported on its own rather than collapsed into one failure.
    """
    surface.state.declaration_conflicts = [
        Conflict(check="experiment_type_vs_trace", declared="a", observed="b")
    ]
    surface.state.window_length = 11
    surface.resolve_conflict(
        "experiment_type_vs_trace",
        reason="stirrer was off for the first hour",
        author="arnold",
    )

    with pytest.raises(Rejected) as error:
        surface.emit_csv("out.csv")
    assert "conflict" not in str(error.value)
    assert "load_run first" in str(error.value)


def test_resolving_without_a_reason_is_refused(surface):
    surface.state.declaration_conflicts = [
        Conflict(check="c", declared="a", observed="b")
    ]
    with pytest.raises(Rejected, match="requires a reason"):
        surface.resolve_conflict("c", reason="  ", author="arnold")


def test_resolving_an_unknown_conflict_is_refused(surface):
    with pytest.raises(Rejected, match="no conflict named"):
        surface.resolve_conflict("nope", reason="because", author="arnold")


def test_a_pending_anomaly_blocks_emit(surface):
    surface.state.window_length = 11
    surface.state.anomalies = [
        AnomalyRecord(
            start_index=10, end_index=12, channel="ls8", detected_by=["derivative"]
        )
    ]
    with pytest.raises(Rejected, match="pending a decision"):
        surface.emit_csv("out.csv")


def test_an_unknown_declaration_key_is_refused(surface):
    with pytest.raises(Rejected, match="not an intake question"):
        surface.set_declaration("colour", "blue", author="arnold")


def test_a_declaration_is_recorded_with_its_derived_values(surface):
    assert surface.set_declaration("instrument", "argen", author="arnold").ok
    assert surface.state.declarations["instrument"].value == "argen"
    assert surface.state.declarations["wavelength_nm"].derived_from == "instrument"


# --- read tier ----------------------------------------------------------------


def test_read_tier_cannot_change_the_run(surface):
    before = surface.state.model_dump_json()
    surface.get_state()
    surface.list_anomalies()
    surface.list_conflicts()
    surface.list_anchor_candidates()
    surface.list_blocked()
    assert surface.state.model_dump_json() == before
