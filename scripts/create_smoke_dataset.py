#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
DEFAULT_NAMES = ['xe_dap', 'xe_may', 'oto_con', 'xe_bus', 'xe_tai', 'xich_lo', 'xe_ba_gac']


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def selected_stems(label_dir: Path, limit: int) -> list[str]:
    labels = sorted(label_dir.glob('*.txt'))
    non_empty = [p for p in labels if p.read_text(errors='replace').strip()]
    empty = [p for p in labels if not p.read_text(errors='replace').strip()]
    picked = non_empty[:limit]
    if len(picked) < limit:
        picked.extend(empty[: limit - len(picked)])
    return [p.stem for p in picked]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, default=Path('data/unified_dataset_fix/unified_dataset_fix'))
    parser.add_argument('--dst', type=Path, default=Path('data/unified_dataset_smoke'))
    parser.add_argument('--train', type=int, default=64)
    parser.add_argument('--val', type=int, default=24)
    parser.add_argument('--test', type=int, default=24)
    args = parser.parse_args()

    limits = {'train': args.train, 'val': args.val, 'test': args.test}
    for split, limit in limits.items():
        src_img = args.src / split / 'images'
        src_lab = args.src / split / 'labels'
        dst_img = args.dst / split / 'images'
        dst_lab = args.dst / split / 'labels'
        stems = selected_stems(src_lab, limit)
        img_by_stem = {p.stem: p for p in src_img.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
        for stem in stems:
            if stem not in img_by_stem:
                raise FileNotFoundError(f'missing image for {split}/{stem}')
            link_or_copy(img_by_stem[stem], dst_img / img_by_stem[stem].name)
            link_or_copy(src_lab / f'{stem}.txt', dst_lab / f'{stem}.txt')
        print(split, len(stems))

    yaml = (
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "\n"
        "nc: 7\n"
        + "names: "
        + repr(DEFAULT_NAMES)
        + "\n"
    )
    (args.dst / 'data.yaml').write_text(yaml, encoding='utf-8')
    print(f'wrote {args.dst / "data.yaml"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
