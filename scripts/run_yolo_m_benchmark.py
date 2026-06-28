#!/usr/bin/env python3
"""Run the YOLO-m baseline/focal experiment matrix.

The script intentionally keeps one dataset, image size, epoch budget, and
pretrained policy across all runs. Focal runs are enabled through environment
variables consumed by the source patches in this workspace.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "experiments/configs/unified_dataset_fix_abs.yaml"
DEFAULT_MATRIX = ROOT / "configs/experiment_matrix.csv"
DEFAULT_PROJECT = ROOT / "experiments/runs"
DONE_MARKER = ".benchmark_done.json"


def read_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def count_result_rows(results_csv: Path) -> int:
    if not results_csv.is_file():
        return 0
    with results_csv.open(encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def run_dir(project: Path, run_id: str) -> Path:
    return project / run_id


def last_checkpoint(project: Path, run_id: str) -> Path:
    return run_dir(project, run_id) / "weights" / "last.pt"


def sidecar_metadata_matches(path: Path, project: Path, data_yaml: Path) -> bool:
    expected = (str(project), str(data_yaml))
    for metadata in (path / "opt.yaml", path / "args.yaml"):
        if not metadata.is_file():
            continue
        text = metadata.read_text(encoding="utf-8", errors="replace")
        if all(item in text for item in expected):
            return True
    return False


def done_marker_matches(path: Path, project: Path, data_yaml: Path, epochs: int) -> bool:
    marker_path = path / DONE_MARKER
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if marker.get("data") != str(data_yaml):
        return False
    if marker.get("project") not in {None, str(project)}:
        return False
    return marker.get("epochs") == epochs


def is_current_run(path: Path, project: Path, data_yaml: Path, epochs: int) -> bool:
    return done_marker_matches(path, project, data_yaml, epochs) or sidecar_metadata_matches(path, project, data_yaml)


def has_completed_run(project: Path, run_id: str, epochs: int, data_yaml: Path) -> bool:
    path = run_dir(project, run_id)
    if not path.exists():
        return False
    has_weight = (path / "weights" / "best.pt").is_file() or (path / "weights" / "last.pt").is_file()
    if not has_weight:
        return False
    if done_marker_matches(path, project, data_yaml, epochs):
        return True
    return sidecar_metadata_matches(path, project, data_yaml) and count_result_rows(path / "results.csv") >= epochs


def resumable_checkpoint(project: Path, run_id: str, epochs: int, data_yaml: Path) -> Path | None:
    ckpt = last_checkpoint(project, run_id)
    if not ckpt.is_file():
        return None
    path = run_dir(project, run_id)
    if is_current_run(path, project, data_yaml, epochs):
        return ckpt
    print(f"\n[{run_id}] found stale checkpoint metadata under {path}; starting fresh in current aiftis project")
    return None


def write_done_marker(project: Path, row: dict[str, str], epochs: int, data_yaml: Path) -> None:
    path = run_dir(project, row["run_id"])
    path.mkdir(parents=True, exist_ok=True)
    marker = {
        "run_id": row["run_id"],
        "model_family": row["model_family"],
        "loss_type": row["loss_type"],
        "epochs": epochs,
        "data": str(data_yaml),
        "project": str(project),
    }
    (path / DONE_MARKER).write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def py_ultralytics_expr(
    model_ref: str,
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    project: Path,
    run_id: str,
    resume_ckpt: Path | None,
) -> str:
    train_ref = str(resume_ckpt) if resume_ckpt else model_ref
    resume_arg = ", resume=True" if resume_ckpt else ""
    return (
        "from ultralytics import YOLO; "
        f"model=YOLO({train_ref!r}); "
        "model.train("
        f"data={str(data_yaml)!r}, epochs={epochs}, imgsz={imgsz}, batch=-1, "
        f"project={str(project)!r}, name={run_id!r}, exist_ok=True, pretrained=True"
        f"{resume_arg}"
        "); "
        f"best={str(project / run_id / 'weights' / 'best.pt')!r}; "
        "model=YOLO(best); "
        "model.val("
        f"data={str(data_yaml)!r}, split='test', imgsz={imgsz}, "
        f"project={str(project)!r}, name={(run_id + '_test')!r}, exist_ok=True"
        ")"
    )

def build_command(
    row: dict[str, str],
    data_yaml: Path,
    epochs: int,
    project: Path,
    resume_ckpt: Path | None = None,
) -> tuple[list[str], Path]:
    family = row["model_family"]
    run_id = row["run_id"]
    source_dir = (ROOT / row["source_dir"]).resolve()
    imgsz = int(row["imgsz"])
    model_ref = row["model_ref"]

    if family == "yolov5":
        if resume_ckpt:
            return ([sys.executable, "train.py", "--resume", str(resume_ckpt)], source_dir)
        return (
            [
                sys.executable,
                "train.py",
                "--img",
                str(imgsz),
                "--batch-size",
                "-1",
                "--epochs",
                str(epochs),
                "--data",
                str(data_yaml),
                "--weights",
                model_ref,
                "--project",
                str(project),
                "--name",
                run_id,
                "--exist-ok",
            ],
            source_dir,
        )

    if family in {"yolov8", "yolo11"}:
        return (
            [
                sys.executable,
                "-c",
                py_ultralytics_expr(model_ref, data_yaml, epochs, imgsz, project, run_id, resume_ckpt),
            ],
            source_dir,
        )

    if family == "yolov9":
        train_script = "train_dual.py" if (source_dir / "train_dual.py").exists() else "train.py"
        if resume_ckpt:
            return ([sys.executable, train_script, "--resume", str(resume_ckpt)], source_dir)
        model_name = Path(model_ref).stem
        cfg = f"models/detect/{model_name}.yaml"
        hyp = "data/hyps/hyp.scratch-high.yaml"
        return (
            [
                sys.executable,
                train_script,
                "--workers",
                "8",
                "--device",
                "0",
                "--batch",
                "16",
                "--data",
                str(data_yaml),
                "--img",
                str(imgsz),
                "--cfg",
                cfg,
                "--weights",
                model_ref,
                "--name",
                run_id,
                "--hyp",
                hyp,
                "--min-items",
                "0",
                "--epochs",
                str(epochs),
                "--project",
                str(project),
                "--exist-ok",
            ],
            source_dir,
        )

    raise ValueError(f"Unsupported model_family: {family}")

def focal_env(row: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    source_dir = (ROOT / row["source_dir"]).resolve()
    env["PYTHONPATH"] = str(source_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["YOLO_EXPERIMENT_RUN_ID"] = row["run_id"]
    env["YOLO_EXPERIMENT_LOSS_TYPE"] = row["loss_type"]
    if row["loss_type"] == "focal_cls":
        env["YOLO_FOCAL_CLS"] = "1"
        env["YOLO_FOCAL_GAMMA"] = row["focal_gamma"] or "2.0"
        env["YOLO_FOCAL_ALPHA"] = row["focal_alpha"] or "0.25"
    else:
        env.pop("YOLO_FOCAL_CLS", None)
        env.pop("YOLO_FOCAL_GAMMA", None)
        env.pop("YOLO_FOCAL_ALPHA", None)
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--run", default="all", help="run_id from matrix, or all")
    parser.add_argument("--smoke", action="store_true", help="Override each run to 1 epoch and write under experiments/runs/smoke")
    parser.add_argument("--no-resume", action="store_true", help="Disable automatic resume from existing last.pt checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args = parser.parse_args()

    matrix = read_matrix(args.matrix)
    selected = [r for r in matrix if args.run == "all" or r["run_id"] == args.run]
    if not selected:
        raise SystemExit(f"No experiment matched --run={args.run}")

    project = args.project.resolve()
    if args.smoke:
        project = project / "smoke"

    for row in selected:
        epochs = 1 if args.smoke else int(row["epochs"])
        data_yaml = args.data.resolve()
        run_id = row["run_id"]

        if not args.no_resume and has_completed_run(project, run_id, epochs, data_yaml):
            print(f"\n[{run_id}] already completed; skipping training")
            continue

        resume_ckpt = None if args.no_resume else resumable_checkpoint(project, run_id, epochs, data_yaml)

        cmd, cwd = build_command(row, data_yaml, epochs, project, resume_ckpt)
        env = focal_env(row)

        print(f"\n[{run_id}] cwd={cwd}")
        if resume_ckpt:
            print(f"resume checkpoint: {resume_ckpt}")
        if row["loss_type"] == "focal_cls":
            print(f"env YOLO_FOCAL_CLS=1 YOLO_FOCAL_GAMMA={env['YOLO_FOCAL_GAMMA']} YOLO_FOCAL_ALPHA={env['YOLO_FOCAL_ALPHA']}")
        print(" ".join(shlex.quote(part) for part in cmd))

        if args.dry_run:
            continue
        if not cwd.exists():
            raise SystemExit(f"Source directory does not exist: {cwd}")
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        write_done_marker(project, row, epochs, data_yaml)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
