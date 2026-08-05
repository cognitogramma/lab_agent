"""Intake: what the run *is*, declared by a person and never inferred.

Design section 6.0. Before a single number is computed the pipeline has to know
which physical model applies, and that knowledge does not live in the data. A
rising trace is suggestive of aggregation, not proof of it; the same shape comes
out of a dilution run whose sample is aggregating faster than it dilutes.
Processing a run flawlessly under the wrong physical model is the most expensive
mistake available here, because every number that results looks entirely normal.

So the rule is: **where the value is not known, ask for it.** Nothing in this
module guesses. Its whole job is to turn each unknown into a specific question
with its candidate answers and their provenance attached, record the answer with
an author and a timestamp, and report what stays blocked until someone answers.

Three properties make that auditable:

1. A suggestion is *offered*, never applied. Prefilling from the file metadata
   or from a previous run produces a :class:`Suggestion`, which does nothing
   until a person confirms it into a :class:`Declaration`.
2. Where sources disagree, the question carries **both** candidates and the
   source of each, and deliberately has no default. Picking one silently would
   bury a real disagreement inside a plausible number.
3. Unanswered questions do not stall the run. They block precisely the columns
   that depend on them, with the reason in plain language, and everything else
   proceeds. Blocked is an acceptable outcome; guessed is not.

There are no model calls here and there never will be. A declaration is a
statement by a named human about their own experiment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from polyrmc.state import Declaration
from polyrmc.tier0.argen_io import lookup_metadata, parse_concentration

QuestionKind = Literal["choice", "number", "text"]


class Option(BaseModel):
    """One candidate answer, carrying where it came from.

    ``source`` is not decoration. When two sources disagree about a physical
    constant, the researcher needs to see which is the paper and which is a
    value someone typed into the code, because that is the whole basis for
    choosing between them.
    """

    value: Any
    label: str
    source: str


class Suggestion(BaseModel):
    """A prefilled answer awaiting confirmation. Never applied on its own."""

    value: Any
    source: str = Field(..., description="Metadata key, config default, or run id.")
    recalled_from: str | None = Field(
        None, description="Run id, when the suggestion came from a previous run."
    )


class Question(BaseModel):
    """One intake question and everything that depends on it."""

    key: str
    text: str
    kind: QuestionKind
    options: list[Option] = Field(default_factory=list)
    suggestion: Suggestion | None = None
    sets: list[str] = Field(
        default_factory=list, description="Config keys this answer determines."
    )
    blocks: list[str] = Field(
        default_factory=list, description="Outputs unavailable while unanswered."
    )
    required: bool = True
    note: str = ""
    disputed: bool = Field(
        False,
        description="Sources disagree. Such a question is offered with no default, "
        "so the disagreement has to be settled by a person rather than by "
        "whichever source happened to be read last.",
    )
    minimum: float | None = None
    maximum: float | None = None

    def validate_answer(self, value: Any) -> Any:
        """Check an answer against this question, or say exactly what is wrong.

        The rejection is the caller's error-recovery signal, so it states what
        failed and what would be acceptable instead.
        """
        if self.kind == "choice":
            allowed = [option.value for option in self.options]
            if value not in allowed:
                raise ValueError(
                    f"{value!r} is not a valid answer to {self.key!r}. "
                    f"Valid answers: {allowed}."
                )
            return value

        if self.kind == "number":
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{self.key!r} takes a number; got {value!r}."
                ) from error
            if self.minimum is not None and number < self.minimum:
                raise ValueError(
                    f"{self.key!r} = {number:g} is below the minimum {self.minimum:g}."
                )
            if self.maximum is not None and number > self.maximum:
                raise ValueError(
                    f"{self.key!r} = {number:g} is above the maximum {self.maximum:g}."
                )
            return number

        text = str(value).strip()
        if not text:
            raise ValueError(f"{self.key!r} cannot be empty.")
        return text


class InstrumentProfile(BaseModel):
    """Fixed, undisputed facts about an instrument.

    Only values that are properties of the hardware itself live here. The
    Rayleigh ratio of toluene is a property of *toluene at a wavelength* and is
    asked for separately, because the sources in this project disagree about it.
    """

    key: str
    label: str
    wavelength_nm: float
    angle_deg: float
    n_channels: int


INSTRUMENTS: dict[str, InstrumentProfile] = {
    "argen": InstrumentProfile(
        key="argen",
        label="Fluence/Yokogawa ARGEN (16 cells, SLS, 90 deg)",
        wavelength_nm=660.0,
        angle_deg=90.0,
        n_channels=16,
    ),
    "nanobrook_omni": InstrumentProfile(
        key="nanobrook_omni",
        label="Brookhaven NanoBrook Omni (DLS)",
        wavelength_nm=640.0,
        angle_deg=90.0,
        n_channels=1,
    ),
}

# Refractive index increment by sample class, cm^3/g. Both from the paper.
DN_DC_BY_CLASS: dict[str, float] = {
    "protein_peptide": 0.185,
    "polysaccharide": 0.150,
}

# Rayleigh ratio of toluene, cm^-1, as a list of candidates per wavelength.
#
# The 660 nm entry has two candidates that disagree by about 26 percent, and
# that disagreement is real and unresolved: the design specification cites the
# paper for 1.19e-5, while the value carried in polyrmc.config is 9.44e-6.
# Since this constant sets the absolute scale, a wrong choice scales Mw and A2
# by the ratio while leaving A2*Mw and Mw/M0 untouched -- which is exactly the
# kind of error that survives every internal consistency check. It is therefore
# asked, not chosen.
TOLUENE_CANDIDATES: dict[float, list[Option]] = {
    660.0: [
        Option(
            value=1.19e-5,
            label="1.19e-5 cm^-1",
            source="DESIGN.md section 8, attributed to the paper",
        ),
        Option(
            value=9.44e-6,
            label="9.44e-6 cm^-1",
            source="polyrmc.config.TOLUENE_RAYLEIGH_RATIO",
        ),
    ],
    640.0: [
        Option(
            value=1.34e-5,
            label="1.34e-5 cm^-1",
            source="DESIGN.md section 8, attributed to the paper",
        ),
    ],
    633.0: [
        Option(
            value=1.087e-5,
            label="1.087e-5 cm^-1",
            source="polyrmc.config.TOLUENE_RAYLEIGH_RATIO",
        ),
    ],
}

class IntakeForm(BaseModel):
    """The questions for one run, with suggestions filled in but not applied."""

    questions: list[Question]

    def by_key(self, key: str) -> Question:
        for question in self.questions:
            if question.key == key:
                return question
        raise KeyError(
            f"{key!r} is not an intake question. Known keys: "
            f"{[q.key for q in self.questions]}"
        )

    def unanswered(self, declarations: dict[str, Declaration]) -> list[Question]:
        """Required questions still waiting on a person."""
        return [
            question
            for question in self.questions
            if question.required and question.key not in declarations
        ]

    def blocked_reasons(self, declarations: dict[str, Declaration]) -> list[str]:
        """Plain-language reasons each output cannot be computed yet.

        Written for ``RunState.blocked``, so a reader who was not present at
        intake can still see what is missing and who to ask.
        """
        reasons: list[str] = []
        for question in self.unanswered(declarations):
            if not question.blocks:
                continue
            targets = ", ".join(question.blocks)
            reasons.append(
                f"{targets}: blocked because {question.key!r} has not been declared "
                f"({question.text})"
            )
        return reasons


def build_intake_form(
    metadata: dict[str, str] | None = None,
    declarations: dict[str, Declaration] | None = None,
    recalled: dict[str, Suggestion] | None = None,
) -> IntakeForm:
    """Build the question set for one run.

    The form is regenerated as answers arrive, because some questions depend on
    earlier ones: the toluene candidates depend on the instrument's wavelength,
    and the dilution geometry is only asked about on a dilution run.

    Parameters
    ----------
    metadata:
        Header metadata from the raw file. Used only to prefill suggestions.
    declarations:
        Answers so far. Determines which conditional questions apply.
    recalled:
        Suggestions recovered from previous runs, keyed by question. These are
        offered for confirmation and cite the run they came from; they are never
        applied silently.
    """
    metadata = metadata or {}
    declared = declarations or {}
    recalled = recalled or {}

    questions: list[Question] = [
        Question(
            key="technique",
            text="Technique -- SLS, DLS, or both?",
            kind="choice",
            options=[
                Option(value="SLS", label="Static light scattering", source="section 6.0"),
                Option(value="DLS", label="Dynamic light scattering", source="section 6.0"),
                Option(value="both", label="Both", source="section 6.0"),
            ],
            sets=["technique"],
            blocks=["k_D"],
            note="k_D requires the DLS path; an SLS-only run cannot produce it.",
        ),
        Question(
            key="instrument",
            text="Which instrument produced this file?",
            kind="choice",
            options=[
                Option(value=profile.key, label=profile.label, source="LAB.md")
                for profile in INSTRUMENTS.values()
            ],
            sets=["wavelength_nm", "angle_deg"],
            blocks=["I_rayleigh", "K"],
        ),
        Question(
            key="sample_class",
            text="Sample class -- protein/peptide, or polysaccharide?",
            kind="choice",
            options=[
                Option(
                    value="protein_peptide",
                    label="Protein or peptide (dn/dc = 0.185 cm^3/g)",
                    source="paper",
                ),
                Option(
                    value="polysaccharide",
                    label="Polysaccharide (dn/dc = 0.150 cm^3/g)",
                    source="paper",
                ),
            ],
            sets=["dn_dc"],
            blocks=["K", "Mw", "A2"],
        ),
        Question(
            key="experiment_type",
            text=(
                "Experiment type -- automatic continuous dilution, or a "
                "fixed-concentration aggregation run?"
            ),
            kind="choice",
            options=[
                Option(
                    value="acd_dilution",
                    label="ACD dilution: concentration decays as c0*exp(-Qt/V)",
                    source="section 6.0",
                ),
                Option(
                    value="fixed_conc_aggregation",
                    label="Fixed concentration: c is constant, Mw/M0 = I_R(t)/I_R(0)",
                    source="section 6.0",
                ),
            ],
            sets=["experiment_type"],
            blocks=["concentration", "Mw_over_M0"],
            note=(
                "OPEN-1. This MUST NOT be inferred from the shape of the trace. A "
                "rising ls8 is suggestive of aggregation, not conclusive: a dilution "
                "run whose sample aggregates faster than it dilutes rises too."
            ),
        ),
        Question(
            key="c0_g_per_cm3",
            text="Initial concentration c0, in g/cm^3?",
            kind="number",
            minimum=0.0,
            sets=["c_0"],
            blocks=["concentration"],
        ),
        Question(
            key="buffer",
            text="Buffer and ionic strength?",
            kind="text",
            required=False,
            note=(
                "Context only. Used for similarity filtering and for the lab record; "
                "never enters a computation."
            ),
        ),
        Question(
            key="ls_units",
            text=(
                "Is the scattering channel a detector count rate/voltage, or already "
                "normalised?"
            ),
            kind="choice",
            options=[
                Option(
                    value="count_rate",
                    label="Count rate or voltage -- I_R recoverable by toluene calibration",
                    source="section 6.0",
                ),
                Option(
                    value="normalized",
                    label="Already normalised -- absolute scale lost",
                    source="section 6.0",
                ),
            ],
            sets=["ls8_units"],
            blocks=["I_rayleigh"],
            note=(
                "OPEN-2. The observed [0.027, 1.0] range on the reference run confirms "
                "the channel is not already absolute I_R, which would be about 1e-5 "
                "cm^-1. It does not distinguish the two options above. Needs someone "
                "with direct knowledge of the acquisition."
            ),
        ),
        Question(
            key="saturation_ceiling",
            text="What is this detector's full-scale reading?",
            kind="number",
            minimum=0.0,
            options=[
                Option(
                    value=1.0,
                    label="1.0 -- the maximum observed on the reference run",
                    source="observed data, NOT an instrument specification",
                ),
            ],
            sets=["saturation_ceiling"],
            blocks=["saturation masking"],
            disputed=True,
            note=(
                "Must be the real detector or ADC ceiling from the instrument "
                "documentation, never a value read off a trace. A ceiling of exactly "
                "1.0 is what you would see if the channel were normalised rather than "
                "a raw count rate -- suggestive, and explicitly not sufficient to "
                "close OPEN-2. Saturated samples are censored data: the true value is "
                "at least the recorded one, so the ceiling decides which points get "
                "masked out of every computed column."
            ),
        ),
        Question(
            key="buffer_blank",
            text="Buffer blank -- the scattering of buffer alone on this instrument?",
            kind="number",
            minimum=0.0,
            sets=["buffer_blank"],
            blocks=["I_rayleigh"],
            note=(
                "Solvent scattering is a constant that does not aggregate. Leaving it "
                "in drags Mw/M0 toward 1."
            ),
        ),
        Question(
            key="toluene_ref",
            text="Toluene reference -- the toluene reading on this instrument?",
            kind="number",
            minimum=0.0,
            sets=["toluene_ref"],
            blocks=["I_rayleigh"],
        ),
    ]

    # Toluene's Rayleigh ratio depends on wavelength, which the instrument
    # answer fixes. Asked only once the instrument is known, so the candidates
    # shown are the ones that actually apply.
    instrument_key = _declared_value(declared, "instrument")
    profile = INSTRUMENTS.get(str(instrument_key)) if instrument_key else None
    if profile is not None:
        candidates = TOLUENE_CANDIDATES.get(profile.wavelength_nm, [])
        questions.append(
            Question(
                key="r_toluene_cm_inv",
                text=(
                    f"Rayleigh ratio of toluene at {profile.wavelength_nm:g} nm, "
                    "in cm^-1?"
                ),
                kind="number",
                minimum=0.0,
                options=candidates,
                sets=["R_toluene"],
                blocks=["I_rayleigh", "Mw", "A2"],
                disputed=len(candidates) > 1,
                note=(
                    "The two candidates disagree by about 26 percent and both are in "
                    "use in this project. The choice scales Mw and A2 by that ratio; "
                    "A2*Mw and Mw/M0 are unaffected, so no internal check will catch "
                    "the wrong one. Confirm against the instrument's own calibration "
                    "record."
                )
                if len(candidates) > 1
                else "",
            )
        )

    # Dilution geometry only applies to a dilution run.
    if _declared_value(declared, "experiment_type") == "acd_dilution":
        questions.extend(
            [
                Question(
                    key="cell_volume_cm3",
                    text="Cuvette volume V, in cm^3?",
                    kind="number",
                    minimum=0.0,
                    options=[
                        Option(
                            value=2.0,
                            label="2 cm^3",
                            source="DESIGN.md section 8, attributed to the paper",
                        ),
                        Option(
                            value=1.0,
                            label="1 cm^3",
                            source="polyrmc.config.DilutionConfig default",
                        ),
                    ],
                    sets=["V"],
                    blocks=["concentration"],
                    disputed=True,
                    note=(
                        "V sets the Q/V decay rate, so it bends the entire "
                        "concentration axis and with it every point of Kc/I_R vs c. "
                        "The two values in this project differ by a factor of two."
                    ),
                ),
                Question(
                    key="flow_rate_cm3_per_s",
                    text="Solvent flow rate Q, in cm^3/s?",
                    kind="number",
                    minimum=0.0,
                    options=[
                        Option(
                            value=0.001,
                            label="0.001 cm^3/s",
                            source="DESIGN.md section 8, attributed to the paper",
                        ),
                    ],
                    sets=["Q"],
                    blocks=["concentration"],
                ),
            ]
        )

    _attach_suggestions(questions, metadata, recalled)
    return IntakeForm(questions=questions)


def _declared_value(declarations: dict[str, Declaration], key: str) -> Any | None:
    declaration = declarations.get(key)
    return declaration.value if declaration is not None else None


def _attach_suggestions(
    questions: list[Question],
    metadata: dict[str, str],
    recalled: dict[str, Suggestion],
) -> None:
    """Prefill suggestions from the file header and from previous runs.

    A disputed question is left with no suggestion on purpose. Offering a
    default there would let a real disagreement be confirmed away with a
    keystroke.
    """
    for question in questions:
        if question.disputed:
            continue

        if question.key in recalled:
            question.suggestion = recalled[question.key]
            continue

        if question.key == "c0_g_per_cm3":
            hit = lookup_metadata(metadata, "c0_g_per_cm3")
            if hit is None:
                continue
            header_key, raw = hit
            value = parse_concentration(raw)
            if value is None:
                # Number without a recognisable unit. Say so instead of
                # assuming one; the assumption is worth a factor of 1000.
                question.note = (
                    f"{question.note} Header {header_key!r} reads {raw!r}, whose units "
                    "could not be identified, so it is not offered as a default."
                ).strip()
                continue
            question.suggestion = Suggestion(
                value=value, source=f"file metadata {header_key!r} = {raw!r}"
            )

        elif question.key == "instrument":
            hit = lookup_metadata(metadata, "instrument")
            if hit is None:
                continue
            header_key, raw = hit
            for profile in INSTRUMENTS.values():
                if profile.key in "".join(str(raw).lower().split("(")[0].split()):
                    question.suggestion = Suggestion(
                        value=profile.key, source=f"file metadata {header_key!r} = {raw!r}"
                    )
                    break


class DeclarationSet:
    """The answers for one run, validated on the way in.

    Validation lives here rather than in whatever asked the question, so an
    invalid answer is refused the same way no matter who supplied it -- a
    researcher at the prompt, a config file, or a recalled suggestion.
    """

    def __init__(self, form: IntakeForm | None = None) -> None:
        self.form = form or build_intake_form()
        self.declarations: dict[str, Declaration] = {}

    def record(
        self,
        key: str,
        value: Any,
        author: str,
        recalled_from: str | None = None,
    ) -> Declaration:
        """Record one answer, refreshing the form in case it unlocked others."""
        if not str(author).strip():
            raise ValueError(
                f"a declaration of {key!r} needs a named author; "
                "an unattributed declaration is not auditable"
            )
        question = self.form.by_key(key)
        checked = question.validate_answer(value)

        declaration = Declaration(
            value=checked,
            author=str(author).strip(),
            timestamp=datetime.now(timezone.utc),
            recalled_from=recalled_from,
        )
        self.declarations[key] = declaration

        # Some answers determine others outright. Those are recorded too, so the
        # provenance log shows the derived value and what it was derived from,
        # rather than a constant appearing from nowhere.
        for derived_key, derived_value in _derived_from(key, checked).items():
            self.declarations[derived_key] = Declaration(
                value=derived_value,
                author=declaration.author,
                timestamp=declaration.timestamp,
                recalled_from=recalled_from,
                derived_from=key,
            )
        return declaration

    def refresh_form(self, metadata: dict[str, str] | None = None) -> IntakeForm:
        """Rebuild the form against the answers so far."""
        self.form = build_intake_form(metadata=metadata, declarations=self.declarations)
        return self.form

    def value(self, key: str, default: Any = None) -> Any:
        declaration = self.declarations.get(key)
        return declaration.value if declaration is not None else default

    def unanswered(self) -> list[Question]:
        return self.form.unanswered(self.declarations)

    def blocked_reasons(self) -> list[str]:
        return self.form.blocked_reasons(self.declarations)

    @property
    def complete(self) -> bool:
        return not self.unanswered()


def _derived_from(key: str, value: Any) -> dict[str, Any]:
    """Values an answer fixes outright, with no further judgment involved."""
    if key == "instrument":
        profile = INSTRUMENTS[str(value)]
        return {
            "wavelength_nm": profile.wavelength_nm,
            "angle_deg": profile.angle_deg,
        }
    if key == "sample_class":
        return {"dn_dc": DN_DC_BY_CLASS[str(value)]}
    return {}


def render_question(question: Question, index: int | None = None) -> str:
    """Format one question for a terminal prompt."""
    prefix = f"[{index}] " if index is not None else ""
    lines = [f"{prefix}{question.text}"]
    if question.note:
        lines.append(f"    note: {question.note}")
    for option in question.options:
        lines.append(f"    - {option.value!r}: {option.label}  (source: {option.source})")
    if question.disputed:
        lines.append(
            "    sources disagree, so there is no default here -- this one has to be "
            "settled by a person"
        )
    elif question.suggestion is not None:
        suggestion = question.suggestion
        origin = suggestion.recalled_from or suggestion.source
        lines.append(f"    suggested: {suggestion.value!r}  (from {origin}, unconfirmed)")
    if question.blocks:
        lines.append(f"    blocks while unanswered: {', '.join(question.blocks)}")
    return "\n".join(lines)


__all__ = [
    "DN_DC_BY_CLASS",
    "INSTRUMENTS",
    "TOLUENE_CANDIDATES",
    "DeclarationSet",
    "InstrumentProfile",
    "IntakeForm",
    "Option",
    "Question",
    "Suggestion",
    "build_intake_form",
    "render_question",
]
