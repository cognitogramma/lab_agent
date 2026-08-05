"""The intake command end to end: no prompting, no model, no API key."""

from __future__ import annotations

import json

import pytest

from polyrmc.cli import main

ANSWERS = {
    "technique": "SLS",
    "instrument": "argen",
    "sample_class": "protein_peptide",
    "experiment_type": "acd_dilution",
    "c0_g_per_cm3": 0.0015,
    "ls_units": "count_rate",
    "saturation_ceiling": 65535,
    "buffer_blank": 0.0,
    "toluene_ref": 1.0,
    "r_toluene_cm_inv": 1.19e-5,
    "cell_volume_cm3": 1.0,
    "flow_rate_cm3_per_s": 0.001,
}


@pytest.fixture
def answers_file(tmp_path):
    def write(overrides: dict | None = None):
        path = tmp_path / "answers.json"
        payload = {**ANSWERS, **(overrides or {})}
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    return write


def test_matching_declarations_exit_zero(virial_argen_file, answers_file, capsys):
    code = main(
        [
            "intake",
            "--raw", str(virial_argen_file),
            "--channel", "ls8",
            "--author", "tester",
            "--answers", answers_file(),
        ]
    )
    assert code == 0
    assert "CONFLICT" not in capsys.readouterr().out


def test_a_mislabelled_run_exits_nonzero_and_says_why(
    virial_argen_file, answers_file, capsys
):
    """The file is a dilution trajectory; declaring it fixed-concentration must not pass."""
    code = main(
        [
            "intake",
            "--raw", str(virial_argen_file),
            "--channel", "ls8",
            "--author", "tester",
            "--answers", answers_file({"experiment_type": "fixed_conc_aggregation"}),
        ]
    )
    assert code == 2
    output = capsys.readouterr().out
    assert "CONFLICT experiment_type_vs_trace" in output
    assert "block emit_csv" in output


def test_declarations_round_trip_through_the_written_file(
    virial_argen_file, answers_file, tmp_path
):
    path = tmp_path / "run.decl.json"
    main(
        [
            "intake",
            "--raw", str(virial_argen_file),
            "--channel", "ls8",
            "--author", "tester",
            "--answers", answers_file(),
            "--declarations", str(path),
        ]
    )
    stored = json.loads(path.read_text(encoding="utf-8"))

    assert stored["experiment_type"]["value"] == "acd_dilution"
    assert stored["experiment_type"]["author"] == "tester"
    assert stored["experiment_type"]["timestamp"]
    # Derived values record what fixed them, rather than appearing from nowhere.
    assert stored["wavelength_nm"]["derived_from"] == "instrument"

    # Re-reading the file supplies the same answers without asking again.
    assert main(
        [
            "intake",
            "--raw", str(virial_argen_file),
            "--channel", "ls8",
            "--author", "tester",
            "--declarations", str(path),
        ]
    ) == 0


def test_answers_with_a_byte_order_mark_are_readable(
    virial_argen_file, tmp_path
):
    """PowerShell's Out-File writes one, and this is a Windows lab."""
    path = tmp_path / "bom.json"
    path.write_text(json.dumps(ANSWERS), encoding="utf-8-sig")

    assert main(
        [
            "intake",
            "--raw", str(virial_argen_file),
            "--channel", "ls8",
            "--author", "tester",
            "--answers", str(path),
        ]
    ) == 0


def test_an_unattributed_declaration_is_refused(virial_argen_file, answers_file):
    with pytest.raises(SystemExit):
        main(
            [
                "intake",
                "--raw", str(virial_argen_file),
                "--channel", "ls8",
                "--author", "",
                "--answers", answers_file(),
            ]
        )


def test_unanswered_questions_are_listed_with_what_they_block(
    virial_argen_file, capsys
):
    code = main(
        [
            "intake",
            "--raw", str(virial_argen_file),
            "--channel", "ls8",
            "--author", "tester",
            "--no-prompt",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "concentration, Mw_over_M0: blocked because 'experiment_type'" in output
    assert "settled by a person" in output
