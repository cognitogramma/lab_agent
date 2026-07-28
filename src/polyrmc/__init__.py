"""PolyRMC light-scattering processing pipeline.

An auditable analysis pipeline for automatic continuous dilution (ACD)
light-scattering data.

Architecture invariant -- **models judge, they never compute**:

    tier2/  orchestrator          agentic, human-in-the-loop
    tier1/  bounded eval loops    propose (deterministic) -> judge (model)
    tier0/  measurement layer     fully local math, zero model dependency

No function in :mod:`polyrmc.tier0` may accept a model client. That is
checked by ``tests/test_tier0_purity.py``.
"""

__version__ = "0.1.0"
