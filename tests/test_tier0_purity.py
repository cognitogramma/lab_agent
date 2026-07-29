"""The architecture invariant, enforced.

Tier 0 has zero external dependency. A model outage, a bad response, or a
provider change can cost a suboptimal smoothing window inside an
already-validated safe band. It must never be able to produce a wrong molar
mass.

This is the grep the design promises would catch a violation in code review.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TIER0 = Path(__file__).resolve().parent.parent / "src" / "polyrmc" / "tier0"

FORBIDDEN_MODULES = {
    "anthropic",
    "langchain",
    "langchain_anthropic",
    "langchain_core",
    "langchain_google_genai",
    "langgraph",
    "openai",
    "google",
    "google_genai",
    "cohere",
    "httpx",
    "requests",
    "urllib",
    "socket",
}

CLIENT_LIKE_PARAMETERS = {
    "client",
    "model",
    "judge",
    "llm",
    "chat",
    "agent",
    "api_key",
    "anthropic",
}


def tier0_modules() -> list[Path]:
    return sorted(TIER0.glob("*.py"))


def test_tier0_has_modules_to_check():
    assert tier0_modules(), "tier 0 should not be empty"


@pytest.mark.parametrize("path", tier0_modules(), ids=lambda p: p.name)
def test_no_model_provider_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    offending = imported & FORBIDDEN_MODULES
    assert not offending, (
        f"{path.name} imports {sorted(offending)}; tier 0 must have no external "
        "or model-provider dependency"
    )


@pytest.mark.parametrize("path", tier0_modules(), ids=lambda p: p.name)
def test_no_function_accepts_a_model_client(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = node.args
        names = [
            argument.arg
            for argument in [
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                *([arguments.vararg] if arguments.vararg else []),
                *([arguments.kwarg] if arguments.kwarg else []),
            ]
        ]
        for name in names:
            if name.lower() in CLIENT_LIKE_PARAMETERS:
                violations.append(f"{node.name}({name})")

    assert not violations, (
        f"{path.name} has model-client-shaped parameters: {violations}. No tier 0 "
        "function may accept a model client."
    )


def test_tier0_imports_only_from_tier0_and_config():
    """Tier 0 must not reach upward into the judging or orchestration layers."""
    for path in tier0_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                module = node.names[0].name
            if module and module.startswith("polyrmc"):
                assert not module.startswith(("polyrmc.tier1", "polyrmc.tier2")), (
                    f"{path.name} imports {module}; tier 0 must not depend on a "
                    "higher tier"
                )
