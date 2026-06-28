#!/usr/bin/env python3
"""Validate a YOLO detection dataset and emit issue reports.

Default policy matches this project: empty label files are allowed as
background/negative samples, but missing labels, orphan labels, malformed
labels, invalid bbox coordinates, and duplicated label rows are failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def parse_names(data_yaml: Path) -> list[str]:
    """Parse the simple data.yaml format used by this dataset."""
    text = data_yaml.read_text(encoding="utf-8")
    names_match = re.search(r"names\s*:\s*(\[.*?\])", text, flags=re.S)
    if names_match:
        raw = names_match.group(1).strip()[1:-1]
        return [item.strip().strip("'\"") for item in raw.split(",") if item.strip()]

    lines = text.splitlines()
    names: list[str] = []
    in_names = False
    for line in lines:
        if line.strip().startswith("names:"):
            in_names = True
            continue
        if in_names:
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith("-"):
                break
            names.append(stripped[1:].strip().strip("'\""))
    return names


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def issue_row(
    dataset_root: Path,
    split: str,
    issue_type: str,
    severity: str,
    path: Path | None = None,
    label_path: Path | None = None,
    image_path: Path | None = None,
    line_no: int | str = "",
    duplicate_of_line: int | str = "",
    class_id: int | str = "",
    class_name: str = "",
    x_center: float | str = "",
    y_center: float | str = "",
    width: float | str = "",
    height: float | str = "",
    details: str = "",
    recommended_action: str = "",
) -> dict[str, object]:
    return {
        "split": split,
        "issue_type": issue_type,
        "severity": severity,
        "status": "todo",
        "path": rel(path, dataset_root) if path else "",
        "label_path": rel(label_path, dataset_root) if label_path else "",
        "image_path": rel(image_path, dataset_root) if image_path else "",
        "line_no": line_no,
        "duplicate_of_line": duplicate_of_line,
        "class_id": class_id,
        "class_name": class_name,
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
        "details": details,
        "recommended_action": recommended_action,
    }


def validate_split(dataset_root: Path, split: str, names: list[str], allow_empty: bool) -> tuple[dict, list[dict]]:
    img_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"
    images = sorted(p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    labels = sorted(p for p in label_dir.rglob("*.txt") if p.is_file())
    img_stems = {p.stem: p for p in images}
    label_stems = {p.stem: p for p in labels}

    issues: list[dict] = []
    class_counts: Counter[int] = Counter()
    source_images: Counter[str] = Counter()
    source_objects: Counter[str] = Counter()
    obj_per_image: list[int] = []
    empty_labels = 0
    duplicate_lines = 0
    invalid_lines = 0

    def source(stem: str) -> str:
        match = re.match(r"(DS\d+)", stem)
        return match.group(1) if match else "unknown"

    for img in images:
        source_images[source(img.stem)] += 1

    for stem in sorted(set(img_stems) - set(label_stems)):
        issues.append(
            issue_row(
                dataset_root,
                split,
                "missing_label",
                "fix",
                path=img_stems[stem],
                image_path=img_stems[stem],
                details="Image has no matching .txt label file.",
                recommended_action="Create an empty label for a valid background image, or annotate missing objects.",
            )
        )

    for stem in sorted(set(label_stems) - set(img_stems)):
        issues.append(
            issue_row(
                dataset_root,
                split,
                "orphan_label",
                "fix",
                path=label_stems[stem],
                label_path=label_stems[stem],
                details="Label has no matching image file.",
                recommended_action="Restore the matching image or remove the orphan label.",
            )
        )

    for label in labels:
        image_path = img_stems.get(label.stem)
        raw_lines = label.read_text(encoding="utf-8", errors="replace").splitlines()
        non_empty = [(i, line.strip()) for i, line in enumerate(raw_lines, 1) if line.strip()]
        object_count = 0

        if not non_empty:
            empty_labels += 1
            if not allow_empty:
                issues.append(
                    issue_row(
                        dataset_root,
                        split,
                        "empty_label",
                        "review",
                        path=label,
                        label_path=label,
                        image_path=image_path,
                        details="Label is empty.",
                        recommended_action="Confirm it is a background image or annotate missing objects.",
                    )
                )

        seen: dict[str, int] = {}
        for line_no, text in non_empty:
            if text in seen:
                duplicate_lines += 1
                parts = text.split()
                cls_text = parts[0] if parts else ""
                cls_name = names[int(cls_text)] if cls_text.isdigit() and int(cls_text) < len(names) else ""
                coords = (parts + ["", "", "", ""])[1:5]
                issues.append(
                    issue_row(
                        dataset_root,
                        split,
                        "duplicate_label_line",
                        "review",
                        path=label,
                        label_path=label,
                        image_path=image_path,
                        line_no=line_no,
                        duplicate_of_line=seen[text],
                        class_id=cls_text,
                        class_name=cls_name,
                        x_center=coords[0],
                        y_center=coords[1],
                        width=coords[2],
                        height=coords[3],
                        details="The exact same label row appears more than once in this file.",
                        recommended_action="Open the image and remove the duplicate row if it is the same object.",
                    )
                )
            else:
                seen[text] = line_no

            parts = text.split()
            if len(parts) != 5:
                invalid_lines += 1
                issues.append(
                    issue_row(
                        dataset_root,
                        split,
                        "invalid_label_format",
                        "fix",
                        path=label,
                        label_path=label,
                        image_path=image_path,
                        line_no=line_no,
                        details=f"Expected 5 YOLO fields, got {len(parts)}: {text}",
                        recommended_action="Rewrite as: class x_center y_center width height.",
                    )
                )
                continue

            try:
                cls_float = float(parts[0])
                cls = int(cls_float)
                x, y, w, h = [float(v) for v in parts[1:]]
            except ValueError:
                invalid_lines += 1
                issues.append(
                    issue_row(
                        dataset_root,
                        split,
                        "invalid_label_numeric",
                        "fix",
                        path=label,
                        label_path=label,
                        image_path=image_path,
                        line_no=line_no,
                        details=f"Non-numeric YOLO value: {text}",
                        recommended_action="Fix class and coordinates to valid numeric values.",
                    )
                )
                continue

            object_count += 1
            class_counts[cls] += 1
            source_objects[source(label.stem)] += 1

            problems = []
            if cls_float != cls:
                problems.append("class_id_not_integer")
            if cls < 0 or cls >= len(names):
                problems.append("class_out_of_range")
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                problems.append("non_finite_coord")
            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                problems.append("coord_out_of_0_1")
            if w <= 0 or h <= 0:
                problems.append("non_positive_width_or_height")
            if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
                problems.append("bbox_extends_outside_image")

            if problems:
                invalid_lines += 1
                issues.append(
                    issue_row(
                        dataset_root,
                        split,
                        "invalid_bbox_or_class",
                        "fix",
                        path=label,
                        label_path=label,
                        image_path=image_path,
                        line_no=line_no,
                        class_id=cls,
                        class_name=names[cls] if 0 <= cls < len(names) else "",
                        x_center=x,
                        y_center=y,
                        width=w,
                        height=h,
                        details=";".join(problems),
                        recommended_action="Fix or remove this object annotation.",
                    )
                )

        if label.stem in img_stems:
            obj_per_image.append(object_count)

    stats = {
        "images": len(images),
        "labels": len(labels),
        "objects": sum(class_counts.values()),
        "missing_labels": len(set(img_stems) - set(label_stems)),
        "orphan_labels": len(set(label_stems) - set(img_stems)),
        "empty_labels": empty_labels,
        "duplicate_label_lines": duplicate_lines,
        "invalid_label_lines": invalid_lines,
        "objects_per_image_mean": sum(obj_per_image) / len(obj_per_image) if obj_per_image else 0.0,
        "class_counts": {names[i]: class_counts[i] for i in range(len(names))},
        "source_images": dict(sorted(source_images.items())),
        "source_objects": dict(sorted(source_objects.items())),
    }
    return stats, issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/unified_dataset_fix/unified_dataset_fix", type=Path)
    parser.add_argument("--csv-out", default="experiments/results/dataset_validation_issues.csv", type=Path)
    parser.add_argument("--json-out", default="experiments/results/dataset_validation_summary.json", type=Path)
    parser.add_argument("--allow-empty-labels", action="store_true", default=True)
    parser.add_argument("--fail-on-issues", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    names = parse_names(data_root / "data.yaml")
    if not names:
        raise SystemExit(f"Could not parse class names from {data_root / 'data.yaml'}")

    all_issues: list[dict] = []
    summary = {"data_root": str(data_root), "names": names, "splits": {}, "total": Counter()}
    for split in SPLITS:
        stats, issues = validate_split(data_root, split, names, args.allow_empty_labels)
        summary["splits"][split] = stats
        all_issues.extend(issues)
        for key in ("images", "labels", "objects", "missing_labels", "orphan_labels", "empty_labels", "duplicate_label_lines", "invalid_label_lines"):
            summary["total"][key] += stats[key]

    summary["total"] = dict(summary["total"])
    summary["issue_count"] = len(all_issues)

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(issue_row(data_root, "", "", "").keys())
    with args.csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_issues)
    args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_issues and all_issues:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
