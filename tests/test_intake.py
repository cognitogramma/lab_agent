"""Intake: nothing is guessed, and a disputed value has no default."""

from __future__ import annotations

import pytest

from continuum_spectranalytics.intake import (
    DN_DC_BY_CLASS,
    INSTRUMENTS,
    DeclarationSet,
    build_intake_form,
    render_question,
)
from continuum_spectranalytics.tier0.argen_io import lookup_metadata, parse_concentration


def test_experiment_type_is_asked_not_inferred():
    form = build_intake_form()
    question = form.by_key("experiment_type")
    assert question.required
    assert {option.value for option in question.options} == {
        "acd_dilution",
        "fixed_conc_aggregation",
    }
    assert question.suggestion is None
    assert "concentration" in question.blocks


def test_disputed_questions_carry_every_candidate_and_no_default():
    """The whole point: a disagreement must not be resolvable by pressing return."""
    declarations = DeclarationSet()
    declarations.record("instrument", "argen", author="tester")
    form = declarations.refresh_form()

    toluene = form.by_key("r_toluene_cm_inv")
    assert toluene.disputed
    assert toluene.suggestion is None
    values = {option.value for option in toluene.options}
    assert values == {1.19e-5, 9.44e-6}
    # Each candidate says where it came from, which is the only basis for
    # choosing between them.
    assert all(option.source for option in toluene.options)


def test_saturation_ceiling_is_declared_never_taken_from_the_data():
    form = build_intake_form()
    ceiling = form.by_key("saturation_ceiling")
    assert ceiling.disputed
    assert ceiling.suggestion is None
    assert any("NOT an instrument specification" in o.source for o in ceiling.options)


def test_instrument_fixes_wavelength_and_records_that_it_was_derived():
    declarations = DeclarationSet()
    declarations.record("instrument", "argen", author="tester")

    assert declarations.value("wavelength_nm") == INSTRUMENTS["argen"].wavelength_nm
    derived = declarations.declarations["wavelength_nm"]
    assert derived.derived_from == "instrument"
    assert derived.author == "tester"


def test_sample_class_fixes_dn_dc():
    declarations = DeclarationSet()
    declarations.record("sample_class", "protein_peptide", author="tester")
    assert declarations.value("dn_dc") == DN_DC_BY_CLASS["protein_peptide"]


def test_invalid_answers_are_refused_with_the_valid_ones_named():
    declarations = DeclarationSet()
    with pytest.raises(ValueError, match="acd_dilution"):
        declarations.record("experiment_type", "aggregation", author="tester")


def test_a_declaration_needs_a_named_author():
    declarations = DeclarationSet()
    with pytest.raises(ValueError, match="named author"):
        declarations.record("technique", "SLS", author="   ")


def test_dilution_geometry_is_asked_only_on_a_dilution_run():
    declarations = DeclarationSet()
    assert "cell_volume_cm3" not in {q.key for q in declarations.form.questions}

    declarations.record("experiment_type", "acd_dilution", author="tester")
    form = declarations.refresh_form()
    volume = form.by_key("cell_volume_cm3")
    assert volume.disputed
    assert {option.value for option in volume.options} == {1.0, 2.0}


def test_unanswered_questions_block_their_own_columns_only():
    declarations = DeclarationSet()
    declarations.record("experiment_type", "fixed_conc_aggregation", author="tester")
    reasons = declarations.blocked_reasons()

    assert not any("experiment_type" in reason for reason in reasons)
    assert any("I_rayleigh" in reason for reason in reasons)


def test_c0_is_suggested_from_the_header_when_the_unit_is_known():
    form = build_intake_form(metadata={"Sample Concentration": "1.5 mg/mL"})
    question = form.by_key("c0_g_per_cm3")
    assert question.suggestion is not None
    assert question.suggestion.value == pytest.approx(1.5e-3)
    assert "Sample Concentration" in question.suggestion.source


def test_c0_is_not_suggested_when_the_unit_is_missing():
    """A bare number could be mg/mL or g/L -- a factor of a thousand in Mw."""
    form = build_intake_form(metadata={"Sample Concentration": "1.5"})
    question = form.by_key("c0_g_per_cm3")
    assert question.suggestion is None
    assert "could not be identified" in question.note


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1.5 mg/mL", 1.5e-3),
        ("0.002 g/cm3", 0.002),
        ("2 g/L", 0.002),
        ("1.5", None),
        ("not a number", None),
        ("3 furlongs/hogshead", None),
    ],
)
def test_parse_concentration(text, expected):
    result = parse_concentration(text)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_lookup_metadata_is_insensitive_to_header_spelling():
    hit = lookup_metadata({"Sample  Concentration": "1 mg/mL"}, "c0_g_per_cm3")
    assert hit == ("Sample  Concentration", "1 mg/mL")


def test_rendered_question_shows_sources_and_what_it_blocks():
    declarations = DeclarationSet()
    declarations.record("instrument", "argen", author="tester")
    text = render_question(declarations.refresh_form().by_key("r_toluene_cm_inv"))

    assert "DESIGN.md" in text
    assert "continuum_spectranalytics.config" in text
    assert "settled by a person" in text
    assert "I_rayleigh" in text
