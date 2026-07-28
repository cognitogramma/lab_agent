"""Deterministic classification rules for flagged regions.

Where the computed features are unambiguous, the type is assigned here and no
model is consulted. Genuinely ambiguous regions return ``None`` and are passed
up to a tier-1 classifier -- the model's job is the hard cases, not all of them.
"""

from __future__ import annotations

from polyrmc.state import AnomalyRecord, AnomalyType


def classify_deterministic(record: AnomalyRecord) -> AnomalyType | None:
    """Assign an anomaly type when the features leave no room for judgment.

    Returns ``None`` when the region is ambiguous, which is a request for
    escalation rather than a statement that nothing is wrong.
    """
    features = record.features
    n_points = features.get("n_points", 0.0)
    core_points = features.get("core_points", n_points)
    amplitude_sigma = features.get("amplitude_in_sigma", 0.0)
    level_shift_sigma = abs(features.get("level_shift_in_sigma", 0.0))
    recovers = features.get("recovers_to_baseline", 0.0) >= 1.0
    longest_run_points = features.get("longest_run_points", 0.0)
    variance_ratio = features.get("region_variance_ratio", 0.0)

    # A detector pinned at its rail repeats the same value exactly. Keyed on the
    # absolute run length, not a fraction of the region: neighbouring detectors
    # routinely merge a stuck stretch with its noisy edges, which dilutes the
    # fraction while leaving the physical signature untouched.
    if "saturation_run" in record.detected_by and longest_run_points >= 5:
        return AnomalyType.SATURATION

    # Short, large, and the trace comes back to the trend it was following.
    # Keyed on core_points -- how many samples actually depart from the trend --
    # rather than the flagged width, which is inflated by the detectors' own
    # rolling windows.
    if core_points <= 3 and amplitude_sigma > 6.0 and recovers:
        return AnomalyType.TRANSIENT_SPIKE

    # Long region that leaves the signal at a different level than it found it.
    if n_points > 50 and level_shift_sigma > 3.0 and not recovers:
        return AnomalyType.BASELINE_DRIFT

    # Elevated variance without a coherent level change.
    if variance_ratio > 10.0 and level_shift_sigma < 1.0 and n_points > 10:
        return AnomalyType.STRUCTURAL_NOISE

    return None


def classify_all(records: list[AnomalyRecord]) -> tuple[list[AnomalyRecord], list[AnomalyRecord]]:
    """Split records into deterministically classified and needs-escalation."""
    settled: list[AnomalyRecord] = []
    ambiguous: list[AnomalyRecord] = []
    for record in records:
        anomaly_type = classify_deterministic(record)
        if anomaly_type is None:
            ambiguous.append(record)
        else:
            record.anomaly_type = anomaly_type
            record.classified_by = "deterministic"
            settled.append(record)
    return settled, ambiguous
