"""Tier 2 -- the orchestrator.

Watches for files, classifies experiment type, routes runs, chains them, and
escalates to a human.

This is the most dangerous component in the system and is deliberately the
least built. Misclassify a run and every downstream number is wrong in a way
that looks entirely normal -- there is no residual or error bar that flags it.
It therefore ships requiring human confirmation of experiment type on **every**
run, and earns the right to run unattended only after a documented record of
correct calls on data where the answer is already known.
"""

from polyrmc.tier2.orchestrator import Orchestrator, PendingRun

__all__ = ["Orchestrator", "PendingRun"]
