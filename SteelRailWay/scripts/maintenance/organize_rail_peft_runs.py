#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Organize outputs/rail_peft/<run>/ into grouped subdirectories.

Layout after organizing:
    <run>/
      summary.txt
      summary.csv
      depth_peft_map.json
      stats/reference_stats.pt
      eval/baseline_scores.csv
      cv/fold1/{history.csv,scores.csv,peft_cam4.pth}
      ...
      final/{final_history.csv,final_peft_scores.csv,final_peft_cam4.pth}
      diagnostics/local_eval/*
      diagnostics/forgetting_check/*
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def move_if_exists(src: Path, dst: Path, replacements: dict[str, str], dry_run: bool) -> None:
    if not src.exists():
        return
    ensure_dir(dst.parent)
    src_key = src.as_posix()
    dst_key = dst.as_posix()
    replacements[src_key] = dst_key
    if dry_run or src.resolve() == dst.resolve():
        return
    shutil.move(str(src), str(dst))


def rewrite_text_references(run_dir: Path, replacements: dict[str, str], dry_run: bool) -> None:
    text_files = list(run_dir.rglob("*.txt")) + list(run_dir.rglob("*.csv")) + list(run_dir.rglob("*.json"))
    for path in text_files:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        new_text = text
        for old, new in replacements.items():
            new_text = new_text.replace(old, new)
        if not dry_run and new_text != text:
            path.write_text(new_text, encoding="utf-8")


def organize_run(run_dir: Path, dry_run: bool = False) -> list[tuple[Path, Path]]:
    view_id = None
    for part in run_dir.name.split("_"):
        if part.startswith("cam") and part[3:].isdigit():
            view_id = int(part[3:])
            break
    if view_id is None:
        raise ValueError(f"cannot infer view id from run name: {run_dir.name}")

    stats_dir = ensure_dir(run_dir / "stats")
    eval_dir = ensure_dir(run_dir / "eval")
    cv_dir = ensure_dir(run_dir / "cv")
    final_dir = ensure_dir(run_dir / "final")
    diagnostics_dir = ensure_dir(run_dir / "diagnostics")
    local_eval_dir = ensure_dir(diagnostics_dir / "local_eval")

    replacements: dict[str, str] = {}
    moves: list[tuple[Path, Path]] = []

    def record(src: Path, dst: Path) -> None:
        if src.exists():
            moves.append((src, dst))
            move_if_exists(src, dst, replacements, dry_run)

    record(run_dir / "reference_stats.pt", stats_dir / "reference_stats.pt")
    record(run_dir / "baseline_scores.csv", eval_dir / "baseline_scores.csv")

    for fold_idx in range(1, 9):
        fold_sources = [
            run_dir / f"fold{fold_idx}_history.csv",
            run_dir / f"fold{fold_idx}_scores.csv",
            run_dir / f"fold{fold_idx}_peft_cam{view_id}.pth",
        ]
        if not any(path.exists() for path in fold_sources):
            continue
        fold_dir = ensure_dir(cv_dir / f"fold{fold_idx}")
        record(fold_sources[0], fold_dir / "history.csv")
        record(fold_sources[1], fold_dir / "scores.csv")
        record(fold_sources[2], fold_dir / f"peft_cam{view_id}.pth")

    record(run_dir / "final_history.csv", final_dir / "final_history.csv")
    record(run_dir / "final_peft_scores.csv", final_dir / "final_peft_scores.csv")
    record(run_dir / f"final_peft_cam{view_id}.pth", final_dir / f"final_peft_cam{view_id}.pth")

    for path in run_dir.glob("local_eval_*"):
        record(path, local_eval_dir / path.name)

    forgetting_dir = run_dir / "forgetting_check"
    if forgetting_dir.exists():
        record(forgetting_dir, diagnostics_dir / "forgetting_check")

    rewrite_text_references(run_dir, replacements, dry_run=dry_run)
    return moves


def discover_runs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and "_p1_" in p.name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize outputs/rail_peft run artifacts.")
    parser.add_argument("--root", type=str, default="outputs/rail_peft")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    runs = discover_runs(root)
    total = 0
    for run_dir in runs:
        moves = organize_run(run_dir, dry_run=args.dry_run)
        total += len(moves)
        mode = "Would move" if args.dry_run else "Moved"
        print(f"{mode} {len(moves)} item(s) under {run_dir}")
        for src, dst in moves:
            print(f"  {src} -> {dst}")
    print(f"Total run dirs: {len(runs)}, total moves: {total}")


if __name__ == "__main__":
    main()
