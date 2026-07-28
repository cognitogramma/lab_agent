"""Run a whole ARGEN session through Part 1, one file per cell.

Pairs each sample with its own ``buffer cell N`` blank by reading the declared
``Cell Number`` from each header, runs the ls8 channel through the pipeline,
and writes a summary table alongside the per-run CSVs and provenance sidecars.

    python examples/run_series.py --data-dir "<folder>" --judge static
    python examples/run_series.py --data-dir "<folder>" --judge model

``--judge model`` calls the pinned Claude model for every subjective decision
and requires ANTHROPIC_API_KEY (put it in .env). ``--judge static`` takes the
conservative option deterministically and makes no network calls.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from polyrmc.config import LoopConfig, OpticalConfig, RunConfig, SmoothingConfig
from polyrmc.io_csv import read_processed_csv
from polyrmc.pipeline import run_part1
from polyrmc.provenance import sidecar_path, summarize_loop
from polyrmc.tier0.argen_io import (
    find_declared_header_length,
    load_argen_file,
    parse_header_metadata,
)
from polyrmc.plotting import plot_run, plot_session
from polyrmc.tier0.dilution import plateau_swing
from polyrmc.tier1.judge import ModelJudge, StaticJudge


def read_header(path: Path) -> dict[str, str]:
    """Metadata block of an ARGEN file, without loading the data."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        preamble = [handle.readline() for _ in range(25)]
    declared = find_declared_header_length(preamble)
    if declared is None:
        raise ValueError(f"{path.name} does not declare its header length")
    return parse_header_metadata(preamble[: declared - 1], sep=",")


def blank_level(path: Path, channel: str) -> float:
    """Median blank counts for the analysed channel."""
    blank = load_argen_file(path, channel=channel)
    return float(np.nanmedian(blank.data[channel].to_numpy(dtype=float)))


def build_judge(kind: str, decision: str, loop: LoopConfig):
    if kind == "static":
        return StaticJudge()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "--judge model needs ANTHROPIC_API_KEY. Put it in .env as a single "
            "line 'ANTHROPIC_API_KEY=sk-ant-...', or use --judge static."
        )
    return ModelJudge(decision, config=loop)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--sample-glob", default="*Sample*.csv")
    parser.add_argument("--blank-prefix", default="buffer cell")
    parser.add_argument("--channel", default="ls8")
    parser.add_argument("--judge", choices=["static", "model"], default="static")
    parser.add_argument(
        "--experiment-type",
        choices=["fixed_concentration", "dilution_trajectory"],
        required=True,
        help="Supplied, never inferred: a misclassified run yields numbers that "
        "look normal and are wrong.",
    )
    parser.add_argument("--dn-dc", type=float, default=None,
                        help="Defaults to the file's Refractive Increment header.")
    parser.add_argument("--summary", type=Path, default=Path("data/runs/series_summary.csv"))
    parser.add_argument("--figures", type=Path, default=Path("data/figures"),
                        help="Directory for one figure per run plus a session figure.")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    everything = sorted(args.data_dir.glob(args.sample_glob))
    blanks = {}
    samples = []
    for path in everything:
        if path.name.lower().startswith(args.blank_prefix):
            blanks[read_header(path).get("Cell Number")] = path
        else:
            samples.append(path)

    if not samples:
        raise SystemExit(f"no sample files matched {args.sample_glob!r}")

    rows = []
    for path in samples:
        header = read_header(path)
        cell = header.get("Cell Number")
        blank_path = blanks.get(cell)
        if blank_path is None:
            print(f"[skip] {path.name}: no blank for cell {cell}")
            continue

        dn_dc = args.dn_dc
        if dn_dc is None:
            dn_dc = float(str(header.get("Refractive Increment", "0.15")).split()[0])

        loop = LoopConfig(max_iterations=2)
        config = RunConfig(
            run_id=path.stem.split(" - Sample")[0].replace(" ", "_")[:60],
            source_file=path,
            experiment_type=args.experiment_type,
            channel=args.channel,
            optical=OpticalConfig(wavelength_nm=660.0, angle_deg=90.0,
                                  solvent_refractive_index=1.333, dn_dc=dn_dc),
            smoothing=SmoothingConfig(min_window=5, max_window=601, n_candidates=12),
            loop=loop,
            solvent_blank_counts=blank_level(blank_path, args.channel),
        )

        try:
            state, csv_path = run_part1(
                config,
                judge=build_judge(args.judge, "smoothing_window", loop),
            )
        except SystemExit:
            raise
        except Exception as error:
            # One unprocessable file must not cost the whole session. Record it
            # and carry on; a refusal is a result the summary should carry.
            print(f"\n{config.run_id}  (cell {cell})")
            print(f"  FAILED: {error}")
            rows.append({
                "run_id": config.run_id, "cell": cell,
                "blank_file": blank_path.name, "rows": 0, "excised": 0,
                "unclassified": 0, "window": None, "candidates": 0,
                "used_fallback": True, "judge": args.judge,
                "mw_over_m0_final": None, "plateau_swing": None,
                "n_warnings": 0, "error": str(error)[:200],
            })
            continue
        frame, _meta = read_processed_csv(csv_path)
        mw = frame["mw_over_m0"].to_numpy(dtype=float)
        excess = frame["rayleigh_ratio_cm_inv"].to_numpy(dtype=float)
        window_loop = state.loops["smoothing_window"]

        print(f"\n{config.run_id}  (cell {cell}, blank {blank_path.name})")
        print(f"  {summarize_loop(window_loop)}")
        print(f"  candidates offered: {len(window_loop.candidates_by_iteration[0])}")
        if window_loop.decisions:
            print(f"  judge said: {window_loop.decisions[0].rationale}")
        for warning in state.warnings:
            print(f"  warning: {warning}")
        print(f"  sidecar: {sidecar_path(csv_path).name}")

        figure_path = ""
        if not args.no_figures:
            figure_path = str(
                plot_run(state, config, frame, args.figures / f"{config.run_id}.png")
            )
            print(f"  figure : {Path(figure_path).name}")

        rows.append({
            "run_id": config.run_id,
            "cell": cell,
            "blank_file": blank_path.name,
            "rows": len(frame),
            "excised": state.splice.n_excised,
            "unclassified": sum(1 for r in state.anomalies if r.anomaly_type is None),
            "window": window_loop.resolved_value,
            "candidates": len(window_loop.candidates_by_iteration[0]),
            "used_fallback": window_loop.used_fallback,
            "judge": args.judge,
            "mw_over_m0_final": round(float(np.nanmedian(mw[-200:])), 4),
            "plateau_swing": round(plateau_swing(excess), 4),
            "n_warnings": len(state.warnings),
            "nd_filter_changes": len(state.nd_filter_changes),
            "at_full_scale": int(np.sum(np.abs(state.raw_signal - 1.0) <= 1e-9)),
            "figure": figure_path,
            "csv": str(csv_path),
            "sidecar": str(sidecar_path(csv_path)),
            "error": "",
        })

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list({key for row in rows for key in row})
    order = ["run_id", "cell", "blank_file", "rows", "excised", "unclassified",
             "window", "candidates", "used_fallback", "judge",
             "mw_over_m0_final", "plateau_swing", "nd_filter_changes",
             "at_full_scale", "n_warnings", "figure", "csv", "sidecar", "error"]
    fieldnames = [k for k in order if k in fieldnames]
    with args.summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsummary -> {args.summary}  ({len(rows)} runs)")

    if not args.no_figures:
        try:
            session = plot_session(rows, args.figures / "session_overview.png")
            print(f"session figure -> {session}")
        except ValueError as error:
            print(f"session figure skipped: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
