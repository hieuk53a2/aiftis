#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

select_python() {
  if [ -n "${PYTHON:-}" ]; then
    if command -v "$PYTHON" >/dev/null 2>&1; then
      command -v "$PYTHON"
      return
    fi
    if [ -x "$PYTHON" ]; then
      printf '%s\n' "$PYTHON"
      return
    fi
    echo "PYTHON is set but not executable: $PYTHON" >&2
    exit 1
  fi

  if [ -x "$ROOT/.conda-yolo/bin/python" ]; then
    printf '%s\n' "$ROOT/.conda-yolo/bin/python"
    return
  fi

  PARENT_ROOT="$(cd "$ROOT/.." && pwd)"
  if [ -x "$PARENT_ROOT/.conda-yolo/bin/python" ]; then
    printf '%s\n' "$PARENT_ROOT/.conda-yolo/bin/python"
    return
  fi

  echo "No YOLO Python environment found. Run: bash $ROOT/scripts/setup_yolo_env.sh" >&2
  exit 1
}

write_data_config() {
  local out="$1"
  local data_root="$2"
  mkdir -p "$(dirname "$out")"
  cat > "$out" <<YAML
path: $data_root
train: train/images
val: val/images
test: test/images

nc: 7
names: ['xe_dap', 'xe_may', 'oto_con', 'xe_bus', 'xe_tai', 'xich_lo', 'xe_ba_gac']
YAML
}

PY="$(select_python)"
SMOKE_DATA="$ROOT/experiments/configs/unified_dataset_smoke_abs.yaml"

"$PY" "$ROOT/scripts/create_smoke_dataset.py" --src "$ROOT/data/unified_dataset_fix/unified_dataset_fix" --dst "$ROOT/data/unified_dataset_smoke" --train 64 --val 24 --test 24
write_data_config "$SMOKE_DATA" "$ROOT/data/unified_dataset_smoke"
"$PY" "$ROOT/scripts/validate_yolo_dataset.py" --data-root "$ROOT/data/unified_dataset_smoke" --csv-out "$ROOT/experiments/results/smoke_dataset_validation_issues.csv" --json-out "$ROOT/experiments/results/smoke_dataset_validation_summary.json" --fail-on-issues
"$PY" "$ROOT/scripts/run_yolo_m_benchmark.py" --smoke --data "$SMOKE_DATA" --run all
"$PY" "$ROOT/scripts/collect_yolo_results.py" --runs "$ROOT/experiments/runs/smoke" --out "$ROOT/experiments/results/smoke_benchmark_summary.csv"
