"""Experiment labels: whose data it is, how it was measured, what it is, and
what question it answers.

These are catalogue metadata, not analysis inputs. Nothing here changes a
number the pipeline computes -- the labels exist so a stored run can be found,
grouped, and correctly interpreted months later, and so a run whose sample or
goal contradicts its :class:`~polyrmc.config.RunConfig` fails loudly at
construction rather than quietly producing a plausible wrong answer.

Every vocabulary is closed. Free text is allowed only where the value is
genuinely open -- a company name, a buffer, a sample name -- because a label
that accepts anything cannot be grouped on, and grouping is the entire point.

**Confidentiality is enforced, not annotated.** The provenance sidecar
serializes the whole :class:`~polyrmc.config.RunConfig` to disk beside the
processed CSV, so a client name recorded here would travel with the data.
:meth:`ExperimentLabel.redacted` returns a copy with the identifying fields
removed, and the sidecar writer uses it for confidential runs. The
confidentiality flag itself, and every scientific label, survive redaction:
what the sample was and how it was measured is not the confidential part.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

REDACTED = "[redacted]"


class Confidentiality(str, Enum):
    """Whether the run carries an external party's proprietary information."""

    CONFIDENTIAL = "confidential"
    NON_CONFIDENTIAL = "non_confidential"


class Ownership(BaseModel):
    """Who the data belongs to.

    Non-confidential work is in-house and records the institution. Confidential
    work is someone else's and must name the company and the project, because
    an unattributable confidential run cannot be returned to its owner or
    deleted on request.
    """

    confidentiality: Confidentiality
    institution: str | None = Field(
        None, description="In-house owner, e.g. Tulane. Non-confidential runs."
    )
    company: str | None = Field(
        None, description="External party. Required when confidential."
    )
    project_type: str | None = Field(
        None,
        description="The client's project, e.g. 'liraglutide', 'mRNA lipid "
        "nanoparticles'. Required when confidential.",
    )

    @model_validator(mode="after")
    def _attribution_matches_confidentiality(self) -> Ownership:
        if self.confidentiality is Confidentiality.CONFIDENTIAL:
            missing = [
                name
                for name in ("company", "project_type")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"confidential runs must name {' and '.join(missing)}; an "
                    "unattributable confidential run cannot be returned or deleted "
                    "on request"
                )
        elif self.institution is None:
            raise ValueError("non-confidential runs must name the institution")
        return self

    def redacted(self) -> Ownership:
        """Copy with the client's identity removed, for sidecars that leave the lab.

        The confidentiality flag is kept. Dropping it would make a redacted
        record indistinguishable from a non-confidential one.
        """
        if self.confidentiality is not Confidentiality.CONFIDENTIAL:
            return self.model_copy(deep=True)
        return self.model_copy(
            update={
                "company": REDACTED,
                "project_type": REDACTED,
            },
            deep=True,
        )


class Method(str, Enum):
    """Measurement techniques a sample may have been characterised with."""

    DLS = "dls"
    SLS = "sls"
    VISCOMETRY = "viscometry"
    FLUORIMETRY = "fluorimetry"
    UV_VIS = "uv_vis"
    RAMAN = "raman"
    CIRCULAR_DICHROISM = "circular_dichroism"


# What each method actually yields. A DLS record claiming an SLS observable is
# a transcription error, and the closed vocabulary catches it at construction.
METHOD_OBSERVABLES: dict[Method, frozenset[str]] = {
    Method.DLS: frozenset({"d_h", "g2", "kcps", "diffusion_coefficient"}),
    Method.SLS: frozenset({"i_r", "kc_over_ir"}),
    Method.VISCOMETRY: frozenset({"raw", "reduced_viscosity"}),
    Method.FLUORIMETRY: frozenset({"intensity"}),
    Method.UV_VIS: frozenset({"absorbance"}),
    Method.RAMAN: frozenset({"intensity"}),
    Method.CIRCULAR_DICHROISM: frozenset({"ellipticity"}),
}

# Methods whose reading is meaningless without a solvent/buffer blank measured
# on the same instrument. Recording that the blank exists is part of the label
# because its absence invalidates the measurement, and that fact is not
# recoverable from the trace afterwards.
BLANK_REQUIRED: frozenset[Method] = frozenset(
    {
        Method.FLUORIMETRY,
        Method.UV_VIS,
        Method.RAMAN,
        Method.CIRCULAR_DICHROISM,
    }
)


class MethodLabel(BaseModel):
    """One technique applied to the sample, and what it produced."""

    method: Method
    observables: tuple[str, ...] = Field(
        (), description="Quantities recorded; must belong to the method."
    )
    blank_measured: bool | None = Field(
        None,
        description="Whether a blank was run. Required for the methods in "
        "BLANK_REQUIRED, meaningless for the others.",
    )
    instrument: str | None = None

    @model_validator(mode="after")
    def _observables_and_blank_match_method(self) -> MethodLabel:
        allowed = METHOD_OBSERVABLES[self.method]
        unknown = sorted(set(self.observables) - allowed)
        if unknown:
            raise ValueError(
                f"{self.method.value} does not produce {unknown}; "
                f"expected any of {sorted(allowed)}"
            )
        if self.method in BLANK_REQUIRED and self.blank_measured is None:
            raise ValueError(
                f"{self.method.value} requires a blank; record blank_measured "
                "explicitly rather than leaving it unknown"
            )
        return self


class SampleClass(str, Enum):
    """What kind of thing was measured."""

    SYNTHETIC_POLYMER = "synthetic_polymer"
    RNA = "rna"
    PROTEIN = "protein"
    VACCINE = "vaccine"
    OTHER = "other"


class ProteinSubclass(str, Enum):
    """Structural class, which sets what a scattering result can mean."""

    MONOCLONAL_ANTIBODY = "monoclonal_antibody"
    GLOBULAR = "globular"
    NON_GLOBULAR = "non_globular"


class SampleLabel(BaseModel):
    """The sample and the solution conditions it was measured in.

    Concentration, pH, buffer and temperature are recorded because a scattering
    result is only interpretable against them -- A2 and k_D are properties of
    the macromolecule *and its solvent*, not of the macromolecule alone.
    """

    name: str
    sample_class: SampleClass
    protein_subclass: ProteinSubclass | None = None

    concentration_g_per_cm3: float | None = Field(None, gt=0)
    ph: float | None = Field(None, ge=0, le=14)
    buffer: str | None = None
    temperature_c: float | None = Field(None, gt=-273.15)

    @model_validator(mode="after")
    def _subclass_only_for_proteins(self) -> SampleLabel:
        if (
            self.protein_subclass is not None
            and self.sample_class is not SampleClass.PROTEIN
        ):
            raise ValueError(
                f"protein_subclass is meaningless for {self.sample_class.value}"
            )
        return self


class Goal(str, Enum):
    """The question the run was designed to answer."""

    EQUILIBRIUM = "equilibrium_characterization"
    KINETICS = "kinetics"
    THERMAL = "thermal"
    DIALYSIS = "dialysis"
    ISOCHEMICAL = "isochemical"


# The variable actually swept, per goal. Closed per goal rather than one flat
# list: 'ph' means a dialysis buffer exchange under DIALYSIS and a titration
# under ISOCHEMICAL, and collapsing them would merge two different experiments.
GOAL_MODES: dict[Goal, frozenset[str]] = {
    Goal.EQUILIBRIUM: frozenset({"acd", "manual"}),
    Goal.KINETICS: frozenset({"aggregation", "hydrolysis", "dissolution"}),
    Goal.THERMAL: frozenset({"temperature_scan", "isothermal"}),
    Goal.DIALYSIS: frozenset({"nacl", "guanidine_hcl", "ph"}),
    Goal.ISOCHEMICAL: frozenset({"ionic_strength", "ph", "denaturant"}),
}


class GoalLabel(BaseModel):
    """The experimental question, and how it was posed."""

    goal: Goal
    mode: str = Field(..., description="The swept variable; see GOAL_MODES.")
    notes: str | None = None

    @model_validator(mode="after")
    def _mode_belongs_to_goal(self) -> GoalLabel:
        allowed = GOAL_MODES[self.goal]
        if self.mode not in allowed:
            raise ValueError(
                f"{self.mode!r} is not a mode of {self.goal.value}; "
                f"expected one of {sorted(allowed)}"
            )
        return self

    @property
    def is_continuous_dilution(self) -> bool:
        """True for automatic continuous dilution, the c(t) = c0*exp(-Qt/V) sweep."""
        return self.goal is Goal.EQUILIBRIUM and self.mode == "acd"


class ExperimentLabel(BaseModel):
    """The complete label for one stored run."""

    ownership: Ownership
    methods: tuple[MethodLabel, ...] = Field(..., min_length=1)
    sample: SampleLabel
    goal: GoalLabel

    @model_validator(mode="after")
    def _methods_are_distinct(self) -> ExperimentLabel:
        seen = [label.method for label in self.methods]
        duplicated = sorted({m.value for m in seen if seen.count(m) > 1})
        if duplicated:
            raise ValueError(
                f"each method may be labelled once; got duplicates {duplicated}"
            )
        return self

    def uses(self, method: Method) -> bool:
        """Whether this run was characterised with the given technique."""
        return any(label.method is method for label in self.methods)

    def redacted(self) -> ExperimentLabel:
        """Copy safe to store outside the lab: client identity removed, science kept."""
        return self.model_copy(
            update={"ownership": self.ownership.redacted()}, deep=True
        )


ExperimentType = Literal["dilution_trajectory", "fixed_concentration"]


def experiment_type_for(goal: GoalLabel) -> ExperimentType | None:
    """The :class:`~polyrmc.config.RunConfig` experiment type a goal implies.

    Only ACD pins the type: it *is* the continuous dilution the pipeline models.
    Every other goal is compatible with either acquisition, so this returns None
    rather than guessing -- the config stays the authority, as
    ``RunConfig.experiment_type`` is supplied and never inferred.
    """
    return "dilution_trajectory" if goal.is_continuous_dilution else None
