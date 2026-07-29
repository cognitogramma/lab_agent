"""Re-export processed runs as plain tab-delimited text for KaleidaGraph et al.

The pipeline's own CSV declares its header length and carries a metadata block
above the column row, which is what makes it re-readable without guessing. That
same block is what a plotting package chokes on, so this script writes a second
copy with one header row and nothing else.

Two resolutions are written for every run. The full file is every point. The
decimated file targets a few thousand rows, keeping the minimum *and* maximum of
each bin rather than taking every Nth point -- plain decimation drops the
saturation excursions and short spikes, which on this data are the features most
worth seeing.

Nothing here recomputes a quantity. It reshapes files that already exist.

    python examples/export_for_plotting.py
    python examples/export_for_plotting.py --target-points 8000
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from polyrmc.io_csv import read_processed_csv

# Written in this order; time first so a plotting package offers it as x.
COLUMNS = (
    ("time_h", "elapsed time (h)"),
    ("time_s", "elapsed time (s)"),
    ("raw_ls8", "raw ls8 (counts)"),
    ("smoothed_ls8", "smoothed ls8 (counts)"),
    ("excess_scattering", "excess scattering"),
    ("mw_over_m0", "Mw/M0"),
)


def envelope_decimate(n_points: int, target: int) -> np.ndarray:
    """Row indices that keep each bin's extremes, so spikes survive.

    Taking every Nth row would hide a 9,000-sample saturation excursion behind
    whichever rows happened to land on the stride.
    """
    if n_points <= target:
        return np.arange(n_points)
    n_bins = max(target // 2, 1)
    edges = np.linspace(0, n_points, n_bins + 1, dtype=int)
    return np.array(sorted({0, n_points - 1, *edges[:-1]}), dtype=int)


def keep_extremes(values: np.ndarray, edges: np.ndarray, n_points: int) -> np.ndarray:
    """Indices of the min and max sample inside every bin."""
    kept: set[int] = {0, n_points - 1}
    bounds = list(edges) + [n_points]
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop <= start:
            continue
        chunk = values[start:stop]
        finite = np.where(np.isfinite(chunk))[0]
        if finite.size == 0:
            kept.add(start)
            continue
        kept.add(start + int(finite[int(np.argmin(chunk[finite]))]))
        kept.add(start + int(finite[int(np.argmax(chunk[finite]))]))
    return np.array(sorted(kept), dtype=int)


def write_table(path: Path, header: list[str], rows) -> Path:
    """Tab-delimited, one header row, no comments -- the format KG expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return path


def export_run(csv_path: Path, out_dir: Path, target: int) -> tuple[Path, Path]:
    frame, meta = read_processed_csv(csv_path)
    time_s = frame["time_s"].to_numpy(dtype=float)
    columns = {
        "time_h": time_s / 3600.0,
        "time_s": time_s,
        "raw_ls8": frame["signal_raw"].to_numpy(dtype=float),
        "smoothed_ls8": frame["signal_smoothed"].to_numpy(dtype=float),
        "excess_scattering": frame["rayleigh_ratio_cm_inv"].to_numpy(dtype=float),
        "mw_over_m0": frame["mw_over_m0"].to_numpy(dtype=float),
    }
    header = [label for _name, label in COLUMNS]
    stack = np.column_stack([columns[name] for name, _label in COLUMNS])
    stem = csv_path.name.replace("_ls8.csv", "")

    full = write_table(out_dir / f"{stem}_full.txt", header, stack)

    edges = envelope_decimate(stack.shape[0], target)
    keep = keep_extremes(columns["raw_ls8"], edges, stack.shape[0])
    small = write_table(out_dir / f"{stem}_plot.txt", header, stack[keep])
    return full, small


def export_ladders(summary_csv: Path, out_dir: Path) -> list[Path]:
    """One file per salt ladder: concentration against the endpoint value.

    Gd and NaCl go to separate files on purpose. They are different experiments,
    and a single table invites plotting them on one concentration axis.
    """
    import re

    rows = [r for r in csv.DictReader(summary_csv.open()) if not r.get("error")]
    written: list[Path] = []
    for ladder, match in (("Gd", "gd"), ("NaCl", "nacl")):
        points = []
        for row in rows:
            name = row["run_id"].lower()
            if match not in name:
                continue
            found = re.search(r"(\d+(?:\.\d+)?)\s*mm", name)
            if not found:
                continue
            reliable = (
                float(row.get("plateau_swing") or 0) <= 0.5
                and int(row.get("nd_filter_changes") or 0) == 0
                and int(row.get("at_full_scale") or 0) <= 100
            )
            points.append(
                (
                    float(found.group(1)),
                    float(row["mw_over_m0_final"]),
                    float(row["plateau_swing"]),
                    int(row["at_full_scale"]),
                    1 if reliable else 0,
                )
            )
        if not points:
            continue
        points.sort()
        written.append(
            write_table(
                out_dir / f"ladder_{ladder}.txt",
                [
                    f"{ladder} (mM)",
                    "Mw/M0 final",
                    "plateau swing",
                    "samples at full scale",
                    "reliable (1=yes)",
                ],
                points,
            )
        )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("data/runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/plotting"))
    parser.add_argument("--pattern", default="liraglutide_sample_2*_ls8.csv")
    parser.add_argument(
        "--target-points",
        type=int,
        default=4000,
        help="Approximate row count for the decimated copy.",
    )
    args = parser.parse_args(argv)

    sources = sorted(args.runs_dir.glob(args.pattern))
    if not sources:
        raise SystemExit(f"no processed CSVs matched {args.pattern!r} in {args.runs_dir}")

    for source in sources:
        full, small = export_run(source, args.out_dir, args.target_points)
        n_full = sum(1 for _ in full.open()) - 1
        n_small = sum(1 for _ in small.open()) - 1
        print(f"{source.name}\n  {full.name}  ({n_full:,} rows)\n  {small.name}  ({n_small:,} rows)")

    summary = args.runs_dir / "series_summary.csv"
    if summary.exists():
        rows = list(csv.DictReader(summary.open()))
        write_table(
            args.out_dir / "series_summary.txt",
            list(rows[0].keys()),
            [list(r.values()) for r in rows],
        )
        print(f"\nseries_summary.txt  ({len(rows)} runs)")
        for path in export_ladders(summary, args.out_dir):
            print(f"{path.name}")

    print(f"\n-> {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
