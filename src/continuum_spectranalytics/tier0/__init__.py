"""Tier 0 -- the measurement layer.

Every derivative, noise-floor estimate, calibration constant, and curve fit in
the pipeline lives here and runs locally. **No function in this package may
accept a model client as an argument.** A model outage can cost you a
suboptimal smoothing window inside an already-validated safe band; it must
never be able to produce a wrong molar mass.

The constraint is enforced by ``tests/test_tier0_purity.py``, which fails on
any model-provider import or client-shaped parameter in this package.
"""
