#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Add or import one P1 DepthAffinePEFT task into an append-only CAD sequence."""

# >>> path-bootstrap >>>
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
# <<< path-bootstrap <<<

import argparse
import json
import subprocess
from datetime import datetime

from rail_cad.p1 import add_p1_training_args, run_p1_experiment
from rail_cad.registry import (
    export_active_depth_peft_map,
    init_registry,
    normalize_task_id,
    register_task_step,
    resolve_artifact_path,
    resolve_base_ckpt,
    resolve_registry_root,
    save_registry,
    snapshot_registry,
    task_id_to_view_id,
    update_task_baseline_eval,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Train or import one P1 task into a CAD registry.")
    add_p1_training_args(
        parser,
        description="L1 CAD: add one task-aware per-view DepthAffinePEFT module.",
        ckpt_required=False,
        default_output_root="outputs/rail_cad",
        default_view_id=4,
    )
    parser.add_argument("--cad_root", type=str, required=True,
                        help="CAD root directory")
    parser.add_argument("--sequence_name", type=str, default=None,
                        help="首次创建 registry 时使用；默认取 cad_root 目录名")
    parser.add_argument("--runs_root", type=str, default="outputs/rail_all",
                        help="自动发现 baseline ckpt 时使用")
    parser.add_argument("--base_ckpt", type=str, default=None,
                        help="可选：覆盖自动发现的 baseline ckpt")
    parser.add_argument("--run_backtest", action="store_true", default=True)
    parser.add_argument("--no_run_backtest", action="store_false", dest="run_backtest")
    parser.add_argument("--import_peft_ckpt", type=str, default=None,
                        help="导入模式：已有 final PEFT ckpt")
    parser.add_argument("--import_run_dir", type=str, default=None,
                        help="导入模式：已有 run 目录（用于 summary/baseline 迁移）")
    parser.add_argument("--import_summary_csv", type=str, default=None)
    parser.add_argument("--import_summary_txt", type=str, default=None)
    parser.add_argument("--allow_missing_import_artifacts", action="store_true")
    return parser


def read_summary_csv(path: Path) -> list[dict]:
    import csv

    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def baseline_eval_from_summary(rows: list[dict]) -> dict:
    for row in rows:
        if row.get("run_type") == "baseline":
            return {
                "source": "summary.csv",
                "auroc": float(row["auroc"]) if row.get("auroc") else None,
                "summary_row": row,
            }
    return {}


def build_step_dir(cad_root: Path, step_idx: int, task_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return cad_root / "steps" / f"step{int(step_idx):03d}_task{task_id}_p1_{timestamp}"


def run_backtest_subprocess(args, step_record: dict, registry_path: Path) -> None:
    cmd = [
        sys.executable,
        str(_PROJ_ROOT / "scripts" / "eval" / "eval_cad_sequence.py"),
        "--cad_root",
        str(registry_path),
        "--step_idx",
        str(step_record["step_idx"]),
        "--out_dir",
        str(resolve_artifact_path(step_record["backtest_dir"])),
        "--train_root",
        str(args.train_root),
        "--test_root",
        str(args.test_root),
        "--img_size",
        str(args.img_size),
        "--batch_size",
        str(args.eval_batch_size),
        "--num_workers",
        str(args.num_workers),
        "--device",
        str(args.device),
        "--depth_norm",
        str(args.depth_norm),
        "--precision",
        str(args.precision),
        "--patch_size",
        str(args.patch_size),
        "--patch_stride",
        str(args.patch_stride),
        "--score_source",
        "fusion",
        "--assist_fill",
        "train_mean",
        "--train_sample_ratio",
        str(args.train_sample_ratio),
        "--sampling_mode",
        str(args.sampling_mode),
        "--train_sample_seed",
        str(args.seed),
    ]
    if args.use_patch:
        cmd.append("--use_patch")
    else:
        cmd.append("--no_patch")
    if args.preload:
        cmd.extend(["--preload", "--preload_workers", str(args.preload_workers)])
    if args.channels_last:
        cmd.append("--channels_last")
    else:
        cmd.append("--no_channels_last")
    if args.train_sample_num is not None:
        cmd.extend(["--train_sample_num", str(args.train_sample_num)])
    subprocess.run(cmd, check=True, cwd=str(_PROJ_ROOT))


def main():
    args = build_parser().parse_args()
    if args.task_id is None:
        args.task_id = str(int(args.view_id))
    task_id = normalize_task_id(args.task_id)
    view_id = task_id_to_view_id(task_id, view_id=args.view_id)

    cad_root = resolve_registry_root(args.cad_root)
    registry, registry_path = init_registry(
        cad_root,
        sequence_name=args.sequence_name,
        defaults={"runs_root": args.runs_root},
    )

    base_ckpt = resolve_base_ckpt(task_id, view_id=view_id, base_ckpt=args.base_ckpt, runs_root=args.runs_root)
    args.ckpt = str(base_ckpt)
    args.view_id = int(view_id)
    args.task_id = task_id

    step_idx = len(registry.get("steps", [])) + 1
    step_dir = build_step_dir(cad_root, step_idx, task_id)
    train_dir = step_dir / "train"
    backtest_dir = step_dir / "backtest"

    baseline_eval = {}
    if args.import_peft_ckpt:
        import_peft_ckpt = resolve_artifact_path(args.import_peft_ckpt)
        if not import_peft_ckpt.exists():
            raise FileNotFoundError(f"Import PEFT checkpoint not found: {import_peft_ckpt}")
        import_run_dir = resolve_artifact_path(args.import_run_dir) if args.import_run_dir else import_peft_ckpt.parents[1]
        summary_csv = resolve_artifact_path(args.import_summary_csv) if args.import_summary_csv else import_run_dir / "summary.csv"
        if summary_csv.exists():
            rows = read_summary_csv(summary_csv)
            baseline_eval = baseline_eval_from_summary(rows)
        elif not args.allow_missing_import_artifacts:
            raise FileNotFoundError(f"Import summary CSV not found: {summary_csv}")
        train_dir.mkdir(parents=True, exist_ok=True)
        import_payload = {
            "mode": "import",
            "task_id": task_id,
            "view_id": int(view_id),
            "base_ckpt": str(base_ckpt),
            "import_peft_ckpt": str(import_peft_ckpt),
            "import_run_dir": str(import_run_dir),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(train_dir / "import_metadata.json", "w", encoding="utf-8") as f:
            json.dump(import_payload, f, ensure_ascii=False, indent=2)
        run_dir_for_registry = import_run_dir
        peft_ckpt_for_registry = import_peft_ckpt
    else:
        result = run_p1_experiment(
            args,
            output_dir=train_dir,
            summary_title=f"Cam{view_id} DepthAffinePEFT P1 (CAD task {task_id})",
            emit_depth_peft_map=False,
        )
        baseline_eval = {
            "source": "train_run",
            "auroc": result["baseline_auroc"],
            "cv_auroc_mean": result["cv_auroc_mean"],
            "cv_auroc_std": result["cv_auroc_std"],
        }
        run_dir_for_registry = result["output_dir"]
        peft_ckpt_for_registry = result["final_ckpt"]

    step_record = register_task_step(
        registry,
        task_id=task_id,
        view_id=view_id,
        base_ckpt=base_ckpt,
        run_dir=run_dir_for_registry,
        peft_ckpt=peft_ckpt_for_registry,
        backtest_dir=backtest_dir,
        stage="p1",
        baseline_eval=baseline_eval,
    )
    update_task_baseline_eval(registry, task_id, baseline_eval)
    save_registry(registry, cad_root)
    snapshot_registry(registry, cad_root, step_idx=step_record["step_idx"])
    active_map_path = export_active_depth_peft_map(registry, cad_root)

    print(f"Registry: {registry_path}")
    print(f"Active depth PEFT map: {active_map_path}")
    print(f"Registered task_id={task_id} view_id={view_id}")
    print(f"PEFT: {resolve_artifact_path(step_record['peft_ckpt'])}")

    if args.run_backtest:
        run_backtest_subprocess(args, step_record, registry_path)


if __name__ == "__main__":
    main()
