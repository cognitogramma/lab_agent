"""The output columns, and the branch each one took to get computed.

Several questions in the design are open and cannot be closed by design work --
they need someone with direct knowledge of how a run was acquired. This module
implements **every** branch of each of them, including the branch where the
question has not been answered yet, so that an unanswered question costs a
specific column and a recorded reason rather than stalling the run or, far
worse, becoming an invisible assumption.

Each function returns the column together with a :class:`PathChoice` naming the
branch taken, why, and what that branch cannot produce. Nothing here infers an
answer from the data: the trace is evidence about the sample, not about how the
experiment was set up.

    OPEN-1  experiment_type  ->  acd_dilution | fixed_conc_aggregation | undeclared
    OPEN-2  ls8 units        ->  count_rate | normalized | undeclared

The undeclared branch is a first-class path, not an error case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polyrmc.state import PathChoice


@dataclass
class ColumnResult:
    """One computed column and the branch that produced it."""

    values: np.ndarray
    path: PathChoice

    @property
    def available(self) -> bool:
        """Whether this column itself came out with values.

        Read off the values rather than off ``path.blocked``, because the two
        are not the same question. A fixed-concentration run produces a
        perfectly good concentration column *and* blocks A2, since one point on
        the concentration axis has no slope. Deriving this from the blocked list
        would report that column as unavailable.
        """
        return bool(np.isfinite(self.values).any())


def _blocked(length: int) -> np.ndarray:
    return np.full(int(length), np.nan, dtype=float)


def concentration_column(
    time_s: np.ndarray,
    experiment_type: str | None,
    c0_g_per_cm3: float | None = None,
    flow_rate_cm3_per_s: float | None = None,
    cell_volume_cm3: float | None = None,
) -> ColumnResult:
    """Concentration in g/cm^3, by whichever route the declared run type implies.

    On a dilution run solvent enters at rate Q while liquid leaves at the same
    rate, so volume is fixed and concentration decays as ``c0*exp(-Qt/V)``. On a
    fixed-concentration aggregation run the concentration is simply c0
    throughout -- constant, not absent, which matters because the concentration
    column is the whole interface to the second half of the pipeline.

    Undeclared is its own branch and MUST NOT be guessed from the trace. A
    rising signal is suggestive of aggregation and no more: a dilution run whose
    sample aggregates faster than it dilutes rises too.
    """
    time_s = np.asarray(time_s, dtype=float)
    alternatives = ["acd_dilution", "fixed_conc_aggregation", "undeclared"]

    if experiment_type is None:
        return ColumnResult(
            _blocked(time_s.size),
            PathChoice(
                question="OPEN-1",
                decision="undeclared",
                because=(
                    "experiment_type has not been declared. It is not inferred from "
                    "the shape of the trace: a rising signal is consistent with both "
                    "an aggregation run and a dilution run whose sample aggregates "
                    "faster than it dilutes."
                ),
                alternatives=[a for a in alternatives if a != "undeclared"],
                blocked=["concentration", "Mw", "A2", "A3"],
            ),
        )

    if experiment_type == "acd_dilution":
        missing = [
            name
            for name, value in (
                ("c_0", c0_g_per_cm3),
                ("Q", flow_rate_cm3_per_s),
                ("V", cell_volume_cm3),
            )
            if value is None or value <= 0
        ]
        if missing:
            return ColumnResult(
                _blocked(time_s.size),
                PathChoice(
                    question="OPEN-1",
                    decision="acd_dilution (incomplete)",
                    because=(
                        f"declared as a dilution run, but {', '.join(missing)} "
                        "is missing, and the decay c0*exp(-Qt/V) cannot be built "
                        "without it"
                    ),
                    alternatives=["fixed_conc_aggregation"],
                    blocked=["concentration", "Mw", "A2", "A3"],
                ),
            )
        decay = float(flow_rate_cm3_per_s) / float(cell_volume_cm3)
        return ColumnResult(
            float(c0_g_per_cm3) * np.exp(-decay * time_s),
            PathChoice(
                question="OPEN-1",
                decision="acd_dilution",
                because=(
                    f"declared as a dilution run; c(t) = c0*exp(-Qt/V) with "
                    f"Q/V = {decay:.6g} /s"
                ),
                alternatives=["fixed_conc_aggregation"],
            ),
        )

    if experiment_type == "fixed_conc_aggregation":
        if c0_g_per_cm3 is None or c0_g_per_cm3 <= 0:
            return ColumnResult(
                _blocked(time_s.size),
                PathChoice(
                    question="OPEN-1",
                    decision="fixed_conc_aggregation (incomplete)",
                    because="declared as a fixed-concentration run, but c_0 is missing",
                    alternatives=["acd_dilution"],
                    blocked=["concentration"],
                ),
            )
        return ColumnResult(
            np.full(time_s.size, float(c0_g_per_cm3), dtype=float),
            PathChoice(
                question="OPEN-1",
                decision="fixed_conc_aggregation",
                because=(
                    f"declared as a fixed-concentration run; c is constant at "
                    f"{float(c0_g_per_cm3):.6g} g/cm^3 throughout"
                ),
                alternatives=["acd_dilution"],
                # A single concentration is one point on the Kc/I_R vs c axis, so
                # there is no virial slope to fit -- but Mw/M0 is exactly what
                # this run type exists to measure.
                blocked=["A2", "A3"],
            ),
        )

    raise ValueError(
        f"{experiment_type!r} is not a known experiment type; expected one of "
        f"{alternatives[:2]}"
    )


def rayleigh_column(
    smoothed: np.ndarray,
    ls_units: str | None,
    buffer_blank: float | None = None,
    toluene_ref: float | None = None,
    r_toluene_cm_inv: float | None = None,
    alpha: float | None = None,
) -> ColumnResult:
    """Absolute Rayleigh ratio in cm^-1, where the units allow it.

    ``I_R = ((S - B) / T) * R_toluene``, computed from the smoothed channel,
    which is the calculation substrate for everything downstream.

    The branch depends on what the scattering channel actually is. If it is a
    detector count rate or voltage, the toluene reference fixes the conversion
    to an absolute scale. If it has already been normalised by the instrument,
    the absolute scale is gone and no calibration recovers it -- that is a
    property of the file, not a missing input.
    """
    smoothed = np.asarray(smoothed, dtype=float)
    alternatives = ["count_rate", "normalized", "undeclared"]

    if ls_units is None:
        return ColumnResult(
            _blocked(smoothed.size),
            PathChoice(
                question="OPEN-2",
                decision="undeclared",
                because=(
                    "the units of the scattering channel have not been declared. The "
                    "observed range confirms it is not already an absolute Rayleigh "
                    "ratio, which would be around 1e-5 cm^-1, but does not "
                    "distinguish a count rate from a normalised channel."
                ),
                alternatives=[a for a in alternatives if a != "undeclared"],
                blocked=["I_rayleigh", "Mw", "A2", "A3"],
            ),
        )

    if ls_units == "normalized":
        return ColumnResult(
            _blocked(smoothed.size),
            PathChoice(
                question="OPEN-2",
                decision="normalized",
                because=(
                    "the channel is already normalised, so its absolute scale was "
                    "discarded by the instrument. No toluene calibration recovers it."
                ),
                alternatives=["count_rate"],
                blocked=["I_rayleigh", "Mw", "A2", "A3"],
            ),
        )

    if ls_units == "count_rate":
        blank = 0.0 if buffer_blank is None else float(buffer_blank)

        # alpha is the same quantity as T/R_toluene, since I_R = (S-B)/alpha and
        # I_R = ((S-B)/T)*R_toluene. It is accepted as an alternative spelling
        # because it can also be back-solved from a standard of known molar mass,
        # which is an independent cross-check on the toluene route -- the two
        # agreeing is itself evidence that alpha really is a fixed scale factor.
        if alpha is not None and (toluene_ref is None or r_toluene_cm_inv is None):
            if alpha <= 0:
                raise ValueError("alpha must be positive")
            return ColumnResult(
                (smoothed - blank) / float(alpha),
                PathChoice(
                    question="OPEN-2",
                    decision="count_rate (alpha)",
                    because=(
                        f"the channel is a count rate, calibrated by the scale factor "
                        f"alpha = {float(alpha):.6g}, which is T/R_toluene expressed "
                        "as one number"
                    ),
                    alternatives=["normalized", "count_rate via toluene reference"],
                ),
            )

        missing = [
            name
            for name, value in (
                ("toluene_ref", toluene_ref),
                ("R_toluene", r_toluene_cm_inv),
            )
            if value is None
        ]
        if missing:
            return ColumnResult(
                _blocked(smoothed.size),
                PathChoice(
                    question="OPEN-2",
                    decision="count_rate (uncalibrated)",
                    because=(
                        f"the channel is a count rate, but {', '.join(missing)} "
                        "has not been supplied, so counts cannot be converted to cm^-1"
                    ),
                    alternatives=["normalized", "count_rate via alpha"],
                    blocked=["I_rayleigh", "Mw", "A2", "A3"],
                ),
            )
        if toluene_ref <= 0:
            raise ValueError("toluene_ref must be positive")
        return ColumnResult(
            (smoothed - blank) / float(toluene_ref) * float(r_toluene_cm_inv),
            PathChoice(
                question="OPEN-2",
                decision="count_rate",
                because=(
                    "the channel is a count rate; I_R = ((S - B)/T)*R_toluene with "
                    f"B = {blank:.6g}, T = {float(toluene_ref):.6g}, "
                    f"R_toluene = {float(r_toluene_cm_inv):.6g} cm^-1"
                ),
                alternatives=["normalized"],
            ),
        )

    raise ValueError(
        f"{ls_units!r} is not a known unit declaration; expected one of "
        f"{alternatives[:2]}"
    )


def mw_over_m0_column(
    smoothed: np.ndarray,
    concentration: np.ndarray | None = None,
    i_rayleigh: np.ndarray | None = None,
    buffer_blank: float = 0.0,
    reference_points: int = 50,
    reference: float | None = None,
) -> ColumnResult:
    """Aggregation index, by the best route the available columns permit.

    ``Mw/M0 = (I_R(t)/c(t)) / (I_R(0)/c(0))`` -- the optical constant K cancels,
    which is why this column survives an uncalibrated instrument. On a
    fixed-concentration run c cancels too and it reduces to ``I_R(t)/I_R(0)``.

    Where I_R is blocked, the same ratio is taken over the smoothed channel
    instead. That substitution is valid *only* because a ratio cancels units,
    and it is recorded explicitly rather than left to be inferred from the fact
    that a number came out.

    The blank is subtracted first in every case. Solvent scattering is a
    constant that does not aggregate, so leaving it in drags the ratio toward 1.
    """
    smoothed = np.asarray(smoothed, dtype=float)

    if i_rayleigh is not None and np.isfinite(i_rayleigh).any():
        intensity = np.asarray(i_rayleigh, dtype=float)
        basis = "I_rayleigh"
        substitution: list[str] = []
    else:
        intensity = smoothed - float(buffer_blank)
        basis = "ls8_smoothed excess"
        substitution = [
            "I_rayleigh is blocked, so Mw/M0 was computed from the smoothed channel "
            "instead. This is valid only because the ratio cancels units; it does "
            "not make the run calibrated."
        ]

    finite = intensity[np.isfinite(intensity)]
    if finite.size == 0:
        return ColumnResult(
            _blocked(smoothed.size),
            PathChoice(
                question="OPEN-2",
                decision=f"{basis} (no finite samples)",
                because="the intensity column has no finite samples to normalise against",
                blocked=["Mw_over_M0"],
            ),
        )

    # A supplied reference is the M0 anchor chosen and validated upstream. The
    # median of the opening samples is only the fallback for callers that have
    # not run that selection.
    if reference is None:
        reference = float(np.median(finite[: min(reference_points, finite.size)]))
    reference = float(reference)
    if reference <= 0:
        return ColumnResult(
            _blocked(smoothed.size),
            PathChoice(
                question="OPEN-2",
                decision=f"{basis} (non-positive reference)",
                because=(
                    f"the reference level is {reference:.6g}, not positive: the blank "
                    "meets or exceeds the sample on this channel, so Mw/M0 is "
                    "undefined. Dividing anyway yields a large negative number that "
                    "looks like a result and means nothing."
                ),
                blocked=["Mw_over_M0"],
            ),
        )

    ratio = intensity / reference

    # On a dilution run the concentration axis has to be divided out, or the
    # dilution itself reads as disaggregation.
    if concentration is not None and np.isfinite(concentration).any():
        c = np.asarray(concentration, dtype=float)
        c_finite = c[np.isfinite(c)]
        varies = c_finite.size > 1 and not np.allclose(c_finite, c_finite[0])
        if varies:
            c0 = float(np.median(c_finite[: min(reference_points, c_finite.size)]))
            if c0 > 0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = ratio / (c / c0)
                basis += " / concentration"

    return ColumnResult(
        ratio,
        PathChoice(
            question="OPEN-2",
            decision=f"Mw/M0 from {basis}",
            because=(
                f"normalised to a reference of {reference:.6g} over the first "
                f"{min(reference_points, finite.size)} finite samples"
            ),
            alternatives=substitution,
        ),
    )


__all__ = [
    "ColumnResult",
    "concentration_column",
    "mw_over_m0_column",
    "rayleigh_column",
]
