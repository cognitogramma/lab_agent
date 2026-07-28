"""Command-line entry point.

    python -m polyrmc.cli part1 --config run.json
    python -m polyrmc.cli part2 --config run.json --csv data/runs/<run>.csv
    python -m polyrmc.cli run   --config run.json

The judge defaults to the deterministic :class:`StaticJudge` unless
``--judge model`` is passed and an API key is configured, so the pipeline is
fully runnable offline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from polyrmc.config import RunConfig
from polyrmc.pipeline import run_part1, run_part2
from polyrmc.provenance import summarize_loop
from polyrmc.tier1.judge import ModelJudge, StaticJudge
from polyrmc.tier1.loop import Judge


def load_config(path: str | Path) -> RunConfig:
    """Load a run configuration from JSON."""
    return RunConfig.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def make_judge(kind: str, decision_name: str, config: RunConfig) -> Judge:
    """Build the judge for a loop, refusing to pretend a model call happened."""
    if kind == "static":
        return StaticJudge()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "--judge model requires ANTHROPIC_API_KEY. Set it in .env, or use "
            "--judge static to run the deterministic conservative rule instead."
        )
    return ModelJudge(decision_name, config=config.loop)


def _report(state, label: str) -> None:
    print(f"\n{label}")
    for loop in state.loops.values():
        print(f"  {summarize_loop(loop)}")
    for warning in state.warnings:
        print(f"  warning: {warning}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(prog="polyrmc", description=__doc__)
    parser.add_argument("stage", choices=["part1", "part2", "run"])
    parser.add_argument("--config", required=True, help="Run configuration JSON.")
    parser.add_argument("--csv", help="Processed CSV (part2 only).")
    parser.add_argument("--channel", help="Scattering channel; defaults to the first.")
    parser.add_argument(
        "--judge",
        choices=["static", "model"],
        default="static",
        help="Tier-1 judge. 'static' takes the conservative option without a model.",
    )
    parser.add_argument("--fit-order", type=int, default=1, choices=[1, 2])
    args = parser.parse_args(argv)

    config = load_config(args.config)

    if args.stage in {"part1", "run"}:
        judge = make_judge(args.judge, "smoothing_window", config)
        state, csv_path = run_part1(config, channel=args.channel, judge=judge)
        _report(state, f"part 1 complete -> {csv_path}")
        csv_for_part2 = csv_path
    else:
        if not args.csv:
            parser.error("part2 requires --csv")
        csv_for_part2 = Path(args.csv)

    if args.stage in {"part2", "run"}:
        judge = make_judge(args.judge, "fit_range", config)
        state = run_part2(csv_for_part2, config, judge=judge, fit_order=args.fit_order)
        _report(state, f"part 2 complete <- {csv_for_part2}")
        parameters = state.parameters
        if parameters:
            print(f"\n  A2*Mw = {parameters.a2_mw:.6g} cm^3/g (calibration-free)")
            print(f"  k_D   = {parameters.k_d:.6g}")
            if parameters.calibrated:
                print(f"  Mw    = {parameters.mw:.6g} g/mol")
                print(f"  A2    = {parameters.a2:.6g} cm^3 mol/g^2")
            else:
                print("  Mw, A2: not reported (alpha not calibrated)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
