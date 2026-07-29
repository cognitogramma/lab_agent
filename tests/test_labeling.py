"""Catalogue labels: closed vocabularies, confidentiality, and config agreement."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from polyrmc.config import OpticalConfig, RunConfig
from polyrmc.labeling import (
    REDACTED,
    Confidentiality,
    ExperimentLabel,
    Goal,
    GoalLabel,
    Method,
    MethodLabel,
    Ownership,
    ProteinSubclass,
    SampleClass,
    SampleLabel,
    experiment_type_for,
)
from polyrmc.provenance import build_record


@pytest.fixture
def in_house_label() -> ExperimentLabel:
    """A non-confidential Tulane ACD run -- the pipeline's own reference case."""
    return ExperimentLabel(
        ownership=Ownership(
            confidentiality=Confidentiality.NON_CONFIDENTIAL, institution="Tulane"
        ),
        methods=(
            MethodLabel(method=Method.SLS, observables=("i_r", "kc_over_ir")),
            MethodLabel(method=Method.DLS, observables=("d_h", "diffusion_coefficient")),
        ),
        sample=SampleLabel(
            name="lysozyme",
            sample_class=SampleClass.PROTEIN,
            protein_subclass=ProteinSubclass.GLOBULAR,
            ph=7.0,
            buffer="50 mM phosphate",
            temperature_c=25.0,
        ),
        goal=GoalLabel(goal=Goal.EQUILIBRIUM, mode="acd"),
    )


@pytest.fixture
def client_label() -> ExperimentLabel:
    """A confidential client run, per the labelling notes."""
    return ExperimentLabel(
        ownership=Ownership(
            confidentiality=Confidentiality.CONFIDENTIAL,
            company="Moderna",
            project_type="mRNA lipid nanoparticles",
        ),
        methods=(MethodLabel(method=Method.DLS, observables=("d_h", "kcps")),),
        sample=SampleLabel(name="LNP-004", sample_class=SampleClass.RNA),
        goal=GoalLabel(goal=Goal.KINETICS, mode="aggregation"),
    )


class TestOwnership:
    def test_confidential_run_must_name_company_and_project(self):
        with pytest.raises(ValidationError, match="must name company and project_type"):
            Ownership(confidentiality=Confidentiality.CONFIDENTIAL)

    def test_confidential_run_missing_only_project_says_so(self):
        with pytest.raises(ValidationError, match="must name project_type"):
            Ownership(
                confidentiality=Confidentiality.CONFIDENTIAL, company="Bachem"
            )

    def test_non_confidential_run_must_name_the_institution(self):
        with pytest.raises(ValidationError, match="must name the institution"):
            Ownership(confidentiality=Confidentiality.NON_CONFIDENTIAL)

    def test_redaction_removes_identity_but_keeps_the_flag(self, client_label):
        redacted = client_label.redacted().ownership

        assert redacted.company == REDACTED
        assert redacted.project_type == REDACTED
        # Keeping the flag is the point: a redacted record must stay
        # distinguishable from a genuinely non-confidential one.
        assert redacted.confidentiality is Confidentiality.CONFIDENTIAL

    def test_redaction_leaves_non_confidential_runs_alone(self, in_house_label):
        assert in_house_label.redacted().ownership.institution == "Tulane"

    def test_redaction_keeps_the_science(self, client_label):
        redacted = client_label.redacted()

        assert redacted.sample == client_label.sample
        assert redacted.goal == client_label.goal
        assert redacted.methods == client_label.methods

    def test_redaction_does_not_mutate_the_original(self, client_label):
        client_label.redacted()
        assert client_label.ownership.company == "Moderna"


class TestMethods:
    def test_observable_must_belong_to_its_method(self):
        with pytest.raises(ValidationError, match="does not produce"):
            MethodLabel(method=Method.DLS, observables=("kc_over_ir",))

    def test_blank_requiring_methods_must_record_the_blank(self):
        with pytest.raises(ValidationError, match="requires a blank"):
            MethodLabel(method=Method.UV_VIS, observables=("absorbance",))

    def test_blank_may_be_recorded_as_absent(self):
        label = MethodLabel(method=Method.RAMAN, blank_measured=False)
        assert label.blank_measured is False

    def test_scattering_methods_do_not_demand_a_blank(self):
        assert MethodLabel(method=Method.SLS).blank_measured is None

    def test_a_method_may_be_labelled_only_once(self, in_house_label):
        with pytest.raises(ValidationError, match="duplicates"):
            ExperimentLabel(
                ownership=in_house_label.ownership,
                methods=(
                    MethodLabel(method=Method.DLS, observables=("d_h",)),
                    MethodLabel(method=Method.DLS, observables=("kcps",)),
                ),
                sample=in_house_label.sample,
                goal=in_house_label.goal,
            )

    def test_at_least_one_method_is_required(self, in_house_label):
        with pytest.raises(ValidationError):
            ExperimentLabel(
                ownership=in_house_label.ownership,
                methods=(),
                sample=in_house_label.sample,
                goal=in_house_label.goal,
            )

    def test_uses_reports_technique_membership(self, in_house_label):
        assert in_house_label.uses(Method.DLS)
        assert not in_house_label.uses(Method.RAMAN)


class TestSample:
    def test_protein_subclass_is_rejected_for_non_proteins(self):
        with pytest.raises(ValidationError, match="meaningless for synthetic_polymer"):
            SampleLabel(
                name="PEG-8000",
                sample_class=SampleClass.SYNTHETIC_POLYMER,
                protein_subclass=ProteinSubclass.GLOBULAR,
            )

    def test_ph_is_bounded(self):
        with pytest.raises(ValidationError):
            SampleLabel(name="x", sample_class=SampleClass.OTHER, ph=15.0)


class TestGoal:
    def test_mode_must_belong_to_the_goal(self):
        with pytest.raises(ValidationError, match="not a mode of thermal"):
            GoalLabel(goal=Goal.THERMAL, mode="aggregation")

    def test_ph_is_a_mode_of_two_different_goals(self):
        # Deliberately not collapsed: a dialysis buffer exchange and an
        # isochemical titration are different experiments.
        assert GoalLabel(goal=Goal.DIALYSIS, mode="ph").goal is Goal.DIALYSIS
        assert GoalLabel(goal=Goal.ISOCHEMICAL, mode="ph").goal is Goal.ISOCHEMICAL

    def test_only_acd_implies_an_experiment_type(self):
        assert (
            experiment_type_for(GoalLabel(goal=Goal.EQUILIBRIUM, mode="acd"))
            == "dilution_trajectory"
        )
        assert experiment_type_for(GoalLabel(goal=Goal.EQUILIBRIUM, mode="manual")) is None
        assert experiment_type_for(GoalLabel(goal=Goal.KINETICS, mode="hydrolysis")) is None


class TestRunConfigAgreement:
    def _config(self, label, experiment_type, dilution):
        return RunConfig(
            run_id="labelled",
            source_file="data/raw/x.txt",
            experiment_type=experiment_type,
            labels=label,
            optical=OpticalConfig(dn_dc=0.185),
            dilution=dilution,
        )

    def test_acd_label_agrees_with_a_dilution_trajectory(self, in_house_label, dilution):
        config = self._config(in_house_label, "dilution_trajectory", dilution)
        assert config.labels is not None

    def test_acd_label_contradicting_the_config_is_refused(self, in_house_label):
        with pytest.raises(ValidationError, match="implies experiment_type"):
            self._config(in_house_label, "fixed_concentration", None)

    def test_other_goals_leave_the_config_free(self, client_label):
        config = self._config(client_label, "fixed_concentration", None)
        assert config.experiment_type == "fixed_concentration"

    def test_labels_are_optional(self, dilution):
        assert self._config(None, "dilution_trajectory", dilution).labels is None


class TestSidecarRedaction:
    def test_sidecar_redacts_a_confidential_client_by_default(
        self, run_config, client_label
    ):
        # A fixed_concentration goal, so the label agrees with any config.
        config = run_config.model_copy(update={"labels": client_label})
        record = build_record(_StubState(config), config)

        assert "Moderna" not in json.dumps(record)
        assert record["config"]["labels"]["ownership"]["company"] == REDACTED
        assert record["config"]["labels"]["sample"]["name"] == "LNP-004"

    def test_internal_archive_can_keep_attribution(self, run_config, client_label):
        config = run_config.model_copy(update={"labels": client_label})
        record = build_record(_StubState(config), config, redact_confidential=False)

        assert record["config"]["labels"]["ownership"]["company"] == "Moderna"

    def test_redaction_does_not_mutate_the_caller_config(self, run_config, client_label):
        config = run_config.model_copy(update={"labels": client_label})
        build_record(_StubState(config), config)

        assert config.labels.ownership.company == "Moderna"


class _StubState:
    """Minimal RunState stand-in: build_record only reads these fields."""

    def __init__(self, config):
        from polyrmc.state import RunState

        self._state = RunState(run_id=config.run_id, source_file=str(config.source_file))

    def __getattr__(self, name):
        return getattr(self._state, name)
