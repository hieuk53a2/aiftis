#!/usr/bin/env python3
"""Collect YOLO run metrics into one benchmark CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "experiments/runs"
DEFAULT_MATRIX = ROOT / "configs/experiment_matrix.csv"
DEFAULT_OUT = ROOT / "experiments/results/benchmark_summary.csv"


MAP_KEYS = (
    "metrics/mAP_0.5:0.95",
    "metrics/mAP50-95(B)",
    "metrics/mAP50-95",
    "mAP50-95",
    "metrics/mAP_0.5:0.95(B)",
)
MAP50_KEYS = ("metrics/mAP_0.5", "metrics/mAP50(B)", "metrics/mAP50", "mAP50")
PRECISION_KEYS = ("metrics/precision", "metrics/precision(B)", "precision")
RECALL_KEYS = ("metrics/recall", "metrics/recall(B)", "recall")


def read_matrix(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return {row["run_id"]: row for row in csv.DictReader(f)}


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def first(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    normalized = {k.strip(): v for k, v in row.items()}
    for key in keys:
        val = as_float(normalized.get(key))
        if val is not None:
            return val
    return None


def best_row(results_csv: Path) -> tuple[int, dict[str, str]] | None:
    with results_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    scored = []
    for idx, row in enumerate(rows):
        score = first(row, MAP_KEYS)
        if score is not None:
            scored.append((score, idx, row))
    if scored:
        score, idx, row = max(scored, key=lambda item: item[0])
        return idx, row
    return len(rows) - 1, rows[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    matrix = read_matrix(args.matrix)
    output = []
    for run_id, config in matrix.items():
        run_dir = args.runs / run_id
        candidates = list(run_dir.rglob("results.csv"))
        row = {
            "run_id": run_id,
            "model_family": config["model_family"],
            "loss_type": config["loss_type"],
            "results_csv": "",
            "best_epoch_index": "",
            "precision": "",
            "recall": "",
            "mAP50": "",
            "mAP50-95": "",
            "status": "missing_results",
        }
        if candidates:
            result = best_row(candidates[0])
            if result:
                idx, best = result
                row.update(
                    {
                        "results_csv": str(candidates[0]),
                        "best_epoch_index": idx,
                        "precision": first(best, PRECISION_KEYS),
                        "recall": first(best, RECALL_KEYS),
                        "mAP50": first(best, MAP50_KEYS),
                        "mAP50-95": first(best, MAP_KEYS),
                        "status": "ok",
                    }
                )
        output.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
