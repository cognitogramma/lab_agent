"""LAB.md -- this lab's conventions, loaded whole into every session.

Without it the agent answers from general knowledge of light scattering rather
than from this instrument's constants, calibration values and house
conventions. It still refuses the right things, because the guards live in
code, but it does not know what this lab does -- and it should say so rather
than answer as though it did.
"""

from __future__ import annotations

import pytest

from continuum_spectranalytics.agent.lab import (
    MISSING_NOTICE,
    find_lab_file,
    load_lab_context,
    load_prompt,
    system_prompt_with_lab,
)
from continuum_spectranalytics.state import RunState


def test_prompts_are_files_a_researcher_can_edit():
    """What the agent may and may not do is the reviewable part of this system.

    It lives on disk rather than in a Python string so that changing it does not
    mean changing code -- and so a reviewer can read what was in force.
    """
    system = load_prompt("system.md")
    assert len(system) > 200
    assert load_prompt("orchestrator.md")


def test_the_projects_lab_file_is_found():
    path = find_lab_file(__file__)
    assert path is not None, "LAB.md should be found by walking up from tests/"
    assert path.name == "LAB.md"


def test_the_lab_context_loads_whole_not_in_fragments():
    context, path = load_lab_context(__file__)

    assert path is not None
    # Whole means whole: the unconfirmed values and the house conventions are
    # both present, and a top-k retrieval would have returned only one of them.
    assert "UNCONFIRMED" in context
    assert "House conventions" in context
    assert "R_toluene" in context
    assert len(context) > 1000


def test_a_missing_lab_file_is_reported_not_tolerated(tmp_path):
    context, path = load_lab_context(tmp_path)

    assert path is None
    assert context == MISSING_NOTICE
    assert "do not substitute general knowledge" in context


def test_the_system_prompt_carries_the_lab_context():
    prompt = system_prompt_with_lab("BASE PROMPT", __file__)

    assert prompt.startswith("BASE PROMPT")
    assert "# Lab context" in prompt
    assert "House conventions" in prompt


def test_the_agent_starts_with_real_lab_context(run_config):
    """An agent running without LAB.md still refuses the right things -- the
    guards are in code -- but it does not know this lab's conventions."""
    from continuum_spectranalytics.agent.orchestrator import ApprovalPolicy, Orchestrator
    from continuum_spectranalytics.agent.toolcall import ScriptedProvider
    from continuum_spectranalytics.agent.toolspec import ToolContext
    from continuum_spectranalytics.memory.notes import NoteBook
    from continuum_spectranalytics.tools import ToolSurface

    agent = Orchestrator(
        context=ToolContext(
            surface=ToolSurface(
                state=RunState(run_id="g", source_file="none"), config=run_config
            ),
            notes=NoteBook(),
        ),
        provider=ScriptedProvider([]),
        approvals=ApprovalPolicy(),
    )

    assert agent.lab_path is not None
    assert "House conventions" in agent.system_prompt
