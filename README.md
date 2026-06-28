# AIFTIS YOLO Benchmark

This repository contains a benchmark pipeline for YOLO-m models on a 7-class traffic dataset. The pipeline currently supports:

- YOLOv5m baseline and focal classification loss
- YOLOv8m baseline and focal classification loss
- YOLOv9m baseline and focal classification loss
- YOLO11m baseline and focal classification loss

All scripts are designed to write generated outputs under this repository's `experiments/` directory.

## Project Structure

```text
aiftis/
  configs/
    experiment_matrix.csv          # Experiment matrix
  data/                            # Local dataset, ignored by Git
  experiments/                     # Local training/validation outputs, ignored by Git
  scripts/
    setup_yolo_env.sh              # Create the Python/Conda environment
    run_smoke_benchmark.sh         # Run a quick 1-epoch smoke benchmark
    run_full_benchmark.sh          # Run the full benchmark
    run_yolo_m_benchmark.py        # Main benchmark runner with auto-resume
    collect_yolo_results.py        # Collect metrics into a CSV summary
    validate_yolo_dataset.py       # Validate the YOLO dataset
  sources/
    yolov5/
    yolov8/
    yolov9/
    yolo11/
```

## Requirements

- Linux or WSL/Linux server
- `bash`
- `conda`
- NVIDIA CUDA GPU recommended for the full benchmark
- Enough disk space for the dataset, checkpoints, and outputs under `data/` and `experiments/`

## Clone The Repository

```bash
git clone https://github.com/hieuk53a2/aiftis.git
cd aiftis
```

## Prepare The Dataset

You can download dataset here: 

The full benchmark expects this dataset layout:

```text
data/unified_dataset_fix/unified_dataset_fix/
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
```

Each image must have a YOLO `.txt` label file with the same stem under the matching `labels/` directory. Label rows use this format:

```text
class_id x_center y_center width height
```

You need to extract the dataset for this path to exist:

```text
data/unified_dataset_fix/unified_dataset_fix/train/images
```

Example:

```bash
mkdir -p data
unzip /path/to/unified_dataset_fix.zip -d data/unified_dataset_fix
```

If the extracted directory is one level too deep or too shallow, move it until it matches the required layout above.

## Prepare Pretrained Weights

Some YOLO source `.gitignore` files ignore `*.pt` weights. If the cloned repository does not include pretrained weights, place or download these files at the following paths:

```text
sources/yolov5/yolov5m.pt
sources/yolov8/yolov8m.pt
sources/yolov9/yolov9-m.pt
sources/yolo11/yolo11m.pt
```

YOLOv5, YOLOv8, and YOLO11 can usually download their weights automatically when internet access is available. For YOLOv9, it is safer to prepare `sources/yolov9/yolov9-m.pt` manually to avoid missing-weight errors.

## Set Up The Environment

Create a local Conda environment inside the repository:

```bash
bash scripts/setup_yolo_env.sh
```

By default, the environment is created at:

```text
.conda-yolo/
```

You can also provide a custom environment path:

```bash
bash scripts/setup_yolo_env.sh /path/to/conda-env
```

Activate it manually if needed:

```bash
conda activate ./.conda-yolo
```

Benchmark scripts resolve Python in this order:

1. `PYTHON` environment variable, if set
2. `.conda-yolo/bin/python` inside this repository
3. `.conda-yolo/bin/python` in the parent directory, for compatibility with the current development machine

Example using a specific Python executable:

```bash
PYTHON=/path/to/python bash scripts/run_smoke_benchmark.sh
```

## Validate The Dataset

Run validation separately if you want to check the dataset before training:

```bash
.conda-yolo/bin/python scripts/validate_yolo_dataset.py \
  --data-root data/unified_dataset_fix/unified_dataset_fix \
  --csv-out experiments/results/dataset_validation_issues.csv \
  --json-out experiments/results/dataset_validation_summary.json \
  --fail-on-issues
```

The validator fails on severe issues such as missing labels, orphan labels, invalid label format, or invalid bounding-box coordinates.

## Run The Smoke Benchmark

The smoke benchmark creates a small dataset from the full dataset and runs each experiment for 1 epoch. Use it to verify the environment and source patches before running the full benchmark.

```bash
bash scripts/run_smoke_benchmark.sh
```

Main outputs:

```text
experiments/runs/smoke/
experiments/results/smoke_benchmark_summary.csv
experiments/results/smoke_dataset_validation_summary.json
```

## Run The Full Benchmark

```bash
bash scripts/run_full_benchmark.sh
```

The script will:

1. Generate `experiments/configs/unified_dataset_fix_abs.yaml` for the current local path
2. Validate the dataset
3. Run all experiments listed in `configs/experiment_matrix.csv`
4. Collect metrics into a summary CSV

Main outputs:

```text
experiments/runs/
experiments/results/benchmark_summary.csv
experiments/results/dataset_validation_summary.json
```

## Run A Single Experiment

You can call the runner directly to run one `run_id` from `configs/experiment_matrix.csv`:

```bash
.conda-yolo/bin/python scripts/run_yolo_m_benchmark.py \
  --data experiments/configs/unified_dataset_fix_abs.yaml \
  --run yolov5m_baseline
```

Available `run_id` values include:

```text
yolov5m_baseline
yolov5m_focal_cls
yolov8m_baseline
yolov8m_focal_cls
yolov9m_baseline
yolov9m_focal_cls
yolo11m_baseline
yolo11m_focal_cls
```

Print commands without training:

```bash
.conda-yolo/bin/python scripts/run_yolo_m_benchmark.py \
  --data experiments/configs/unified_dataset_fix_abs.yaml \
  --run all \
  --dry-run
```

## Resume After Power Loss Or Terminal Disconnect

`run_yolo_m_benchmark.py` enables auto-resume by default.

When you rerun the same script:

- Completed runs are skipped.
- Interrupted runs resume from `weights/last.pt` if the checkpoint is valid.
- Checkpoints or metadata from a different project path are ignored to avoid writing outputs outside the current repository.

Disable auto-resume if you want to start over:

```bash
.conda-yolo/bin/python scripts/run_yolo_m_benchmark.py \
  --data experiments/configs/unified_dataset_fix_abs.yaml \
  --run yolov5m_baseline \
  --no-resume
```

Note: if the interruption happens before the first epoch writes `last.pt`, there is no checkpoint to resume from.

When running over SSH, use `tmux` or `screen` so the process keeps running after terminal disconnects:

```bash
tmux new -s aiftis
bash scripts/run_full_benchmark.sh
```

## Results And Metrics

After a benchmark finishes, inspect the summary:

```bash
cat experiments/results/benchmark_summary.csv
```

Individual runs are stored under:

```text
experiments/runs/<run_id>/
```

Typical files include:

```text
weights/best.pt
weights/last.pt
results.csv
opt.yaml or args.yaml
```

