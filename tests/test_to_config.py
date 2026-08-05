"""Declarations are the configuration, not a record kept beside one.

Before ``to_config`` existed, ``experiment_type`` lived in a hand-written config
*and* in the declarations with nothing reconciling them, so a run could be
declared one thing and processed as another -- the exact failure the intake
layer exists to prevent.
"""

from __future__ import annotations

import json

import pytest

from polyrmc.cli import main
from polyrmc.intake import DeclarationSet, MissingDeclarations, to_config

ANSWERS = {
    "technique": "SLS",
    "instrument": "argen",
    "sample_class": "protein_peptide",
    "experiment_type": "acd_dilution",
    "c0_g_per_cm3": 0.004,
    "ls_units": "count_rate",
    "saturation_ceiling": 65535,
    "buffer_blank": 0.0,
    "toluene_ref": 5000.0,
    "r_toluene_cm_inv": 1.19e-5,
    "cell_volume_cm3": 1.0,
    "flow_rate_cm3_per_s": 0.001,
}


def _declare(**overrides):
    declarations = DeclarationSet()
    for key, value in {**ANSWERS, **overrides}.items():
        if value is None:
            continue
        declarations.record(key, value, author="tester")
        declarations.refresh_form()
    return declarations.declarations


def test_the_declared_experiment_type_is_the_one_processed():
    config = to_config(_declare(), source_file="x.txt", run_id="r")
    assert config.experiment_type == "acd_dilution"


def test_the_instrument_fixes_the_optics():
    config = to_config(_declare(), source_file="x.txt", run_id="r")
    assert config.optical.wavelength_nm == 660.0
    assert config.optical.dn_dc == 0.185


def test_the_dilution_geometry_comes_from_the_declaration():
    config = to_config(_declare(), source_file="x.txt", run_id="r")
    assert config.dilution.c0_g_per_cm3 == 0.004
    assert config.dilution.flow_rate_cm3_per_s == 0.001
    assert config.initial_concentration == 0.004


def test_a_fixed_concentration_run_still_carries_c0():
    """It has no dilution geometry, but c0 is what its concentration column is."""
    declared = _declare(
        experiment_type="fixed_conc_aggregation",
        cell_volume_cm3=None,
        flow_rate_cm3_per_s=None,
    )
    config = to_config(declared, source_file="x.txt", run_id="r")

    assert config.dilution is None
    assert config.c0_g_per_cm3 == 0.004
    assert config.initial_concentration == 0.004


def test_the_calibration_declarations_reach_the_config():
    config = to_config(_declare(), source_file="x.txt", run_id="r")
    assert config.ls_units == "count_rate"
    assert config.toluene_ref == 5000.0
    assert config.r_toluene_cm_inv == 1.19e-5
    assert config.detectors.saturation_ceiling == 65535.0


def test_an_undeclared_ceiling_stays_none_rather_than_defaulting():
    declared = _declare(saturation_ceiling=None)
    config = to_config(declared, source_file="x.txt", run_id="r")
    assert config.detectors.saturation_ceiling is None


def test_an_undeclared_run_is_refused_by_name():
    with pytest.raises(MissingDeclarations) as error:
        to_config({}, source_file="x.txt", run_id="r")

    assert "experiment_type" in error.value.missing
    assert "never inferred from the trace" in str(error.value)


def test_a_dilution_run_missing_its_geometry_is_refused():
    declared = _declare(cell_volume_cm3=None, flow_rate_cm3_per_s=None)
    with pytest.raises(MissingDeclarations) as error:
        to_config(declared, source_file="x.txt", run_id="r")

    assert "cell_volume_cm3" in error.value.missing


# --- the end-to-end command ---------------------------------------------------


def _write_declarations(tmp_path, name="run.decl.json", **overrides):
    """Write a declarations file the way ``intake`` does."""
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                key: json.loads(d.model_dump_json())
                for key, d in _declare(**overrides).items()
            }
        ),
        encoding="utf-8",
    )
    return path


def test_intake_then_process_needs_no_hand_written_config(
    virial_argen_file, tmp_path, capsys
):
    """The whole path: declare once, then process. No config file anywhere."""
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps(ANSWERS), encoding="utf-8")
    declarations = tmp_path / "run.decl.json"

    assert (
        main(
            [
                "intake",
                "--raw", str(virial_argen_file),
                "--author", "tester",
                "--answers", str(answers),
                "--declarations", str(declarations),
            ]
        )
        == 0
    )

    out = tmp_path / "out.csv"
    assert (
        main(
            [
                "process",
                "--raw", str(virial_argen_file),
                "--declarations", str(declarations),
                "--out", str(out),
            ]
        )
        == 0
    )
    assert out.exists()

    printed = capsys.readouterr().out
    assert "verification" in printed
    # The declared calibration actually reached the computation, rather than
    # the run falling back to the uncalibrated smoothed channel.
    assert "against I_rayleigh" in printed


def test_process_refuses_an_undeclared_run(virial_argen_file, tmp_path, capsys):
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")

    code = main(
        ["process", "--raw", str(virial_argen_file), "--declarations", str(empty)]
    )
    assert code == 2
    assert "Run intake first" in capsys.readouterr().out


def test_process_refuses_a_run_whose_data_contradicts_its_declaration(
    virial_argen_file, tmp_path, capsys
):
    """The fixture is a dilution trajectory; declaring it fixed-concentration fails."""
    path = _write_declarations(
        tmp_path,
        "bad.decl.json",
        experiment_type="fixed_conc_aggregation",
        cell_volume_cm3=None,
        flow_rate_cm3_per_s=None,
    )
    code = main(
        ["process", "--raw", str(virial_argen_file), "--declarations", str(path)]
    )

    assert code == 2
    printed = capsys.readouterr().out
    assert "CONFLICT experiment_type_vs_trace" in printed
    assert "Refusing to process" in printed


def test_force_processes_over_a_conflict_and_records_it(
    virial_argen_file, tmp_path, capsys
):
    """A researcher may have a physical explanation. It goes on the record."""
    path = _write_declarations(
        tmp_path,
        "bad.decl.json",
        experiment_type="fixed_conc_aggregation",
        cell_volume_cm3=None,
        flow_rate_cm3_per_s=None,
    )
    main(
        [
            "process",
            "--raw", str(virial_argen_file),
            "--declarations", str(path),
            "--out", str(tmp_path / "forced.csv"),
            "--force",
        ]
    )
    printed = capsys.readouterr().out

    assert "CONFLICT" in printed
    assert "explicit --force" in printed
    # The conflict is still unresolved, so verification fails on it.
    assert "FAILED   0. declarations consistent" in printed
    assert "1 check(s) FAILED" in printed
