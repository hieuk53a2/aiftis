#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/sources"

clone_if_missing() {
  local url="$1"
  local dest="$2"
  if [ -d "$dest/.git" ]; then
    echo "exists: $dest"
    git -C "$dest" rev-parse --short HEAD
    return
  fi
  if [ -e "$dest" ]; then
    echo "refusing to overwrite non-git path: $dest" >&2
    exit 1
  fi
  git clone "$url" "$dest"
  git -C "$dest" rev-parse --short HEAD
}

clone_if_missing https://github.com/ultralytics/yolov5.git "$ROOT/sources/yolov5"
clone_if_missing https://github.com/ultralytics/ultralytics.git "$ROOT/sources/yolov8"
clone_if_missing https://github.com/WongKinYiu/yolov9.git "$ROOT/sources/yolov9"
clone_if_missing https://github.com/ultralytics/ultralytics.git "$ROOT/sources/yolo11"
