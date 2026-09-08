#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Organize outputs/rail_all into per-camera folders.

Before:
    outputs/rail_all/20260501_001404_cam2_bs32_lr0.005_img512_ratio1.0/
    outputs/rail_all/eval_summary_20260501_101021.csv

After:
    outputs/rail_all/Cam2/20260501_001404_cam2_bs32_lr0.005_img512_ratio1.0/
    outputs/rail_all/_summaries/eval_summary_20260501_101021.csv
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


RUN_RE = re.compile(r"(^|_)cam([1-8])(_|$)", re.IGNORECASE)
SUMMARY_PREFIXES = ("summary_", "eval_summary_")


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_dup{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def organize(root: Path, dry_run: bool = False) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    root.mkdir(parents=True, exist_ok=True)
    summaries_dir = root / "_summaries"

    for path in sorted(root.iterdir()):
        if path.name.startswith(".") or path.name.startswith("_"):
            continue

        if path.is_dir():
            if re.fullmatch(r"Cam[1-8]", path.name, re.IGNORECASE):
                continue
            match = RUN_RE.search(path.name)
            if not match:
                continue
            cam_dir = root / f"Cam{match.group(2)}"
            dest = unique_destination(cam_dir / path.name)
            moves.append((path, dest))
            continue

        if path.is_file() and path.name.startswith(SUMMARY_PREFIXES):
            dest = unique_destination(summaries_dir / path.name)
            moves.append((path, dest))

    if dry_run:
        return moves

    for src, dest in moves:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))

    return moves


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize rail_all outputs by camera.")
    parser.add_argument("--root", type=str, default="outputs/rail_all")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    moves = organize(Path(args.root), dry_run=args.dry_run)
    mode = "Would move" if args.dry_run else "Moved"
    print(f"{mode} {len(moves)} item(s)")
    for src, dest in moves:
        print(f"{src} -> {dest}")


if __name__ == "__main__":
    main()
