"""Continuum Spectranalytics research workspace and scattering pipeline.

An auditable analysis pipeline for automatic continuous dilution (ACD)
light-scattering data.

Architecture invariant -- **models judge, they never compute**:

    tier2/  orchestrator          agentic, human-in-the-loop
    tier1/  bounded eval loops    propose (deterministic) -> judge (model)
    tier0/  measurement layer     fully local math, zero model dependency

No function in :mod:`continuum_spectranalytics.tier0` may accept a model client. That is
checked by ``tests/test_tier0_purity.py``.
"""

import os

__version__ = "0.2.0"
APP_NAME = "Continuum Spectranalytics"


def env_setting(suffix: str, default: str | None = None) -> str | None:
    """Read the current project setting, falling back to the old prefix."""
    value = os.environ.get(f"CONTINUUM_SPECTRANALYTICS_{suffix}")
    return value if value is not None else os.environ.get(f"POLYRMC_{suffix}", default)
