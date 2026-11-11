"""A run you have not declared anything about.

Declaring a run used to be a gate you passed before you could open it: a
declarations file, written by ``continuum-spectranalytics intake``, or nothing happened. That gate
bought less than it looked like it did. A session started without one was
scaffolded from ``examples/run_config.example.json``, so ``dn_dc`` was silently
0.150 -- and a protein run taken all the way through would have reported a molar
mass a third out, with a clean fit, good residuals and no warning anywhere.

So "undeclared" is a state the configuration holds now, and the gate is gone.
What each answer buys is graded, and the grading is what these tests pin:

* nothing at all       -- load, clean, decide anomalies, sweep, pick, emit
* experiment_type, c0  -- a concentration axis, and so A2*Mw and k_D
* sample_class         -- Mw, A2 and A3 separately

Nothing is guessed to fill a gap, and nothing stalls waiting for an answer it
never needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from continuum_spectranalytics.agent.toolspec import build_registry
from continuum_spectranalytics.cli import open_session
from continuum_spectranalytics.config import OpticalConfig, RunConfig
from continuum_spectranalytics.tier0.optics import UndeclaredConstant, optical_constant
from continuum_spectranalytics.tools import Rejected


def session_args(raw, tmp_path, **over):
    base = dict(
        raw=str(raw) if raw else None, declarations=None, config=None, run_id="undeclared",
        channel=None, runs_dir=str(tmp_path / "runs"), notes_dir=str(tmp_path / "m"),
        out=str(tmp_path / "out.csv"), author="tester",
    )
    base.update(over)
    return SimpleNamespace(**base)


# --- the configuration can hold "nobody has said yet" ------------------------


def test_a_configuration_needs_no_declarations_to_exist():
    config = RunConfig(run_id="bare", source_file="somewhere.txt")
    assert config.experiment_type is None
    assert config.optical.dn_dc is None
    assert config.initial_concentration is None


def test_k_refuses_rather_than_defaulting():
    """The refusal names both values, because that is the whole difficulty."""
    with pytest.raises(UndeclaredConstant) as raised:
        optical_constant(OpticalConfig())

    message = str(raised.value)
    assert "0.185" in message and "0.150" in message
    assert "cannot tell you which" in message


def test_the_two_plausible_answers_are_a_third_apart():
    """Why a default would be indefensible, in one number.

    K goes as (dn/dc)^2 and Mw as 1/K, so the wrong one of two entirely ordinary
    answers does not produce an obviously broken result -- it produces a molar
    mass that is 34% out and looks fine.
    """
    protein = optical_constant(OpticalConfig(dn_dc=0.185))
    polysaccharide = optical_constant(OpticalConfig(dn_dc=0.150))
    assert protein / polysaccharide == pytest.approx(1.52, abs=0.01)
    assert abs(1 - polysaccharide / protein) == pytest.approx(0.34, abs=0.01)


# --- opening a run -----------------------------------------------------------


def test_a_raw_file_on_its_own_opens(argen_file, tmp_path):
    session = open_session(session_args(argen_file, tmp_path))
    assert session.declarations == {}
    assert session.config.optical.dn_dc is None
    assert str(session.config.source_file) == str(argen_file)


def test_nothing_is_ever_scaffolded_from_the_example_configuration(tmp_path):
    """The regression guard for the leak this replaced.

    A session with no run used to be built from the reviewed example config,
    which carries a real dn_dc. Anything that then loaded a file and extracted
    parameters would have used it, silently, for a sample nobody had described.
    """
    session = open_session(session_args(None, tmp_path, run_id="empty"))
    assert session.config.optical.dn_dc is None, (
        "a session with no run must not inherit a physical constant from an example"
    )
    assert session.config.experiment_type is None
    assert session.config.initial_concentration is None


# --- what an undeclared run can actually do ----------------------------------


@pytest.fixture
def undeclared(virial_argen_file, tmp_path):
    session = open_session(session_args(virial_argen_file, tmp_path))
    return session, build_registry()


def test_an_undeclared_run_loads_sweeps_picks_and_emits(undeclared):
    session, registry = undeclared

    registry["load_run"].handler(session.context, {})
    assert session.surface.state.raw_signal is not None

    registry["sweep_windows"].handler(session.context, {})
    band = [int(c.value) for c in session.surface.get_shortlist()]
    assert band, "the safe band needs no optics to compute"

    registry["pick_window"].handler(session.context, {"window_length": band[-1]})
    for record in session.surface.state.anomalies:
        if record.decision == "pending":
            session.surface.set_decision((record.start_index, record.end_index), "keep")

    registry["emit_csv"].handler(session.context, {})
    assert session.context.output_csv.exists()


def test_a_number_out_of_reach_comes_back_as_a_reason(undeclared):
    """Not a stack trace, and not a guess: the question that is still open."""
    session, registry = undeclared
    registry["load_run"].handler(session.context, {})
    registry["sweep_windows"].handler(session.context, {})
    band = [int(c.value) for c in session.surface.get_shortlist()]
    registry["pick_window"].handler(session.context, {"window_length": band[-1]})
    for record in session.surface.state.anomalies:
        if record.decision == "pending":
            session.surface.set_decision((record.start_index, record.end_index), "keep")
    registry["emit_csv"].handler(session.context, {})

    with pytest.raises((ValueError, Rejected)) as raised:
        registry["extract_parameters"].handler(session.context, {})
    assert "concentration" in str(raised.value)


def test_what_is_blocked_says_why_it_is_blocked(undeclared):
    session, registry = undeclared
    registry["load_run"].handler(session.context, {})

    blocked = registry["list_blocked"].handler(session.context, {})
    assert "saturation" in blocked
    # The reason has to explain why the trace cannot answer it either, or the
    # obvious next move is to read the ceiling off the data.
    assert "railed trace is the rail" in blocked


# --- an undeclared dn_dc costs the absolutes and nothing else ----------------


def test_a2_mw_survives_an_undeclared_dn_dc(virial_argen_file, tmp_path, optical, dilution):
    """K is a constant y-scale, and A2*Mw does not see a constant y-scale.

    The same cancellation that makes alpha optional. Refusing to fit at all
    would withhold two correct numbers to punish a missing answer they never
    depended on.
    """
    from continuum_spectranalytics.config import RunConfig
    from continuum_spectranalytics.pipeline import run_part1, run_part2

    declared = RunConfig(
        run_id="declared", source_file=virial_argen_file,
        experiment_type="acd_dilution", optical=optical, dilution=dilution,
    )
    _state, csv = run_part1(declared, output_csv=tmp_path / "declared.csv")

    without = declared.model_copy(
        update={"optical": optical.model_copy(update={"dn_dc": None})}
    )
    a = run_part2(csv, declared, fit_order=1).parameters
    b = run_part2(csv, without, fit_order=1).parameters

    assert b.a2_mw == pytest.approx(a.a2_mw, rel=1e-9), "A2*Mw must not depend on K"
    assert b.k_d == pytest.approx(a.k_d, rel=1e-9)
    assert b.calibrated is False
    assert not b.mw and not b.a2, "the absolutes are withheld, not scaled by a guess"
