#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backtest a CAD registry across all seen tasks and compute continual metrics."""

# >>> path-bootstrap >>>
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
# <<< path-bootstrap <<<

import argparse
import csv
import gc
import json
from datetime import datetime
from types import SimpleNamespace

import torch

from rail_cad.metrics import build_matrix_payload, compute_continual_metrics, write_matrix_csv
from rail_cad.registry import (
    build_active_depth_peft_map,
    list_seen_task_ids,
    load_registry,
    normalize_task_id,
    resolve_artifact_path,
    resolve_base_ckpt,
    resolve_registry_path,
    resolve_active_peft_ckpt,
    task_id_to_view_id,
)
from scripts.eval.eval_from_ckpt import SCORE_SOURCES, evaluate_from_args


def build_parser():
    parser = argparse.ArgumentParser(description="Backtest CAD routed PEFTs on all seen tasks.")
    parser.add_argument("--cad_root", type=str, required=True,
                        help="CAD root directory or registry.json path")
    parser.add_argument("--step_idx", type=int, default=None,
                        help="默认回测当前最新 step；也可指定历史 step")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="默认写到对应 step 的 backtest 目录")
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--test_root", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--depth_norm", type=str, default="zscore", choices=["zscore", "minmax", "log"])
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--use_patch", action="store_true", default=True)
    parser.add_argument("--no_patch", action="store_false", dest="use_patch")
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--preload_workers", type=int, default=8)
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--no_channels_last", action="store_false", dest="channels_last")
    parser.add_argument("--score_source", type=str, default="fusion", choices=SCORE_SOURCES)
    parser.add_argument("--assist_fill", type=str, default="train_mean", choices=["train_mean", "zeros"])
    parser.add_argument("--assist_stats_dir", type=str, default=None)
    parser.add_argument("--assist_stats_batch_size", type=int, default=16)
    parser.add_argument("--train_sample_ratio", type=float, default=1.0)
    parser.add_argument("--train_sample_num", type=int, default=None)
    parser.add_argument("--sampling_mode", type=str, default="uniform_time", choices=["uniform_time", "random"])
    parser.add_argument("--train_sample_seed", type=int, default=42)
    return parser


def base_eval_namespace(args, ckpt: str, view_id: int, result_json: Path):
    return SimpleNamespace(
        ckpt=str(ckpt),
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=int(view_id),
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        depth_norm=args.depth_norm,
        use_patch=args.use_patch,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        preload=args.preload,
        preload_workers=args.preload_workers,
        precision=args.precision,
        channels_last=args.channels_last,
        output_log=None,
        append_log=False,
        scores_csv=None,
        result_json=str(result_json),
        depth_peft_ckpt=None,
        cad_registry=None,
        score_source=args.score_source,
        scores_dir=str(result_json.parent),
        assist_fill=args.assist_fill,
        assist_stats_dir=args.assist_stats_dir,
        assist_stats_batch_size=args.assist_stats_batch_size,
        train_sample_ratio=args.train_sample_ratio,
        train_sample_num=args.train_sample_num,
        sampling_mode=args.sampling_mode,
        train_sample_seed=args.train_sample_seed,
        module_ablation="full",
        fusion_rule="sum",
    )


def format_metric(value) -> str:
    return "" if value is None else f"{float(value):.8f}"


def clone_result_payload(result: dict) -> dict:
    return json.loads(json.dumps(result, ensure_ascii=False))


def run_config(args, *, config_name: str, ckpt: str, view_id: int, result_dir: Path,
               depth_peft_ckpt: str | None = None, cad_registry: str | None = None) -> dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    ns = base_eval_namespace(args, ckpt, view_id, result_dir / "result.json")
    ns.depth_peft_ckpt = depth_peft_ckpt
    ns.cad_registry = cad_registry
    print(f"Evaluating Cam{view_id} / {config_name}")
    if depth_peft_ckpt:
        print(f"  depth_peft_ckpt={depth_peft_ckpt}")
    if cad_registry:
        print(f"  cad_registry={cad_registry}")
    result = evaluate_from_args(ns)
    return clone_result_payload(result)


def build_summary_row(task_id: str, view_id: int, base_ckpt: str, baseline: dict, cad_active: dict, shared_current: dict):
    def auroc_from(payload):
        if not payload:
            return None
        metric_map = payload.get("auroc_by_source") or {}
        value = metric_map.get("fusion")
        if value is None:
            value = payload.get("auroc")
        return value

    baseline_auroc = auroc_from(baseline)
    active_auroc = auroc_from(cad_active)
    shared_auroc = auroc_from(shared_current)
    return {
        "task_id": str(task_id),
        "view_id": int(view_id),
        "base_ckpt": str(base_ckpt),
        "baseline_depth_peft_ckpt": str(baseline.get("depth_peft_ckpt", "")),
        "cad_active_depth_peft_ckpt": str(cad_active.get("depth_peft_ckpt", "")),
        "shared_current_depth_peft_ckpt": str(shared_current.get("depth_peft_ckpt", "")),
        "baseline_auroc": format_metric(baseline_auroc),
        "cad_active_auroc": format_metric(active_auroc),
        "shared_current_auroc": format_metric(shared_auroc),
        "delta_active_vs_baseline": format_metric(None if baseline_auroc is None or active_auroc is None else active_auroc - baseline_auroc),
        "delta_shared_vs_baseline": format_metric(None if baseline_auroc is None or shared_auroc is None else shared_auroc - baseline_auroc),
        "delta_active_vs_shared": format_metric(None if shared_auroc is None or active_auroc is None else active_auroc - shared_auroc),
        "baseline_status": baseline.get("status", ""),
        "cad_active_status": cad_active.get("status", ""),
        "shared_current_status": shared_current.get("status", ""),
    }


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "task_id",
        "view_id",
        "base_ckpt",
        "baseline_depth_peft_ckpt",
        "cad_active_depth_peft_ckpt",
        "shared_current_depth_peft_ckpt",
        "baseline_auroc",
        "cad_active_auroc",
        "shared_current_auroc",
        "delta_active_vs_baseline",
        "delta_shared_vs_baseline",
        "delta_active_vs_shared",
        "baseline_status",
        "cad_active_status",
        "shared_current_status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_txt(path: Path, args, registry: dict, step_idx: int, current_task_id: str, rows: list[dict], metrics_payload: dict) -> None:
    current_metrics = metrics_payload.get("current_step") or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("CAD backtest summary\n")
        f.write(f"created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"registry: {resolve_registry_path(args.cad_root)}\n")
        f.write(f"sequence_name: {registry.get('sequence_name', '')}\n")
        f.write(f"step_idx: {step_idx}\n")
        f.write(f"current_task_id: {current_task_id}\n")
        f.write(f"score_source: {args.score_source}\n")
        f.write(f"train_root: {args.train_root}\n")
        f.write(f"test_root: {args.test_root}\n\n")
        for row in rows:
            f.write(
                f"task_id={row['task_id']} view_id={row['view_id']}: "
                f"baseline={row['baseline_auroc'] or 'N/A'}, "
                f"cad_active={row['cad_active_auroc'] or 'N/A'}, "
                f"shared_current={row['shared_current_auroc'] or 'N/A'}, "
                f"d_active_vs_baseline={row['delta_active_vs_baseline'] or 'N/A'}, "
                f"d_active_vs_shared={row['delta_active_vs_shared'] or 'N/A'}\n"
            )
        if current_metrics:
            f.write("\n")
            f.write(
                "continual_metrics: "
                f"ACC={format_metric(current_metrics.get('acc')) or 'N/A'}, "
                f"BWT={format_metric(current_metrics.get('bwt')) or 'N/A'}, "
                f"avg_forgetting={format_metric(current_metrics.get('avg_forgetting')) or 'N/A'}, "
                f"max_forgetting={format_metric(current_metrics.get('max_forgetting')) or 'N/A'}, "
                f"retention={format_metric(current_metrics.get('retention')) or 'N/A'}, "
                f"FWT={format_metric(current_metrics.get('fwt')) or 'N/A'}\n"
            )


def main():
    args = build_parser().parse_args()
    registry = load_registry(args.cad_root)
    if not registry.get("steps"):
        raise RuntimeError("CAD registry has no steps to backtest")

    step_idx = int(args.step_idx) if args.step_idx is not None else int(registry["steps"][-1]["step_idx"])
    step_record = None
    for item in registry["steps"]:
        if int(item["step_idx"]) == step_idx:
            step_record = item
            break
    if step_record is None:
        raise KeyError(f"step_idx not found in registry: {step_idx}")

    current_task_id = str(step_record["task_id"])
    current_peft_ckpt = resolve_artifact_path(step_record["peft_ckpt"])
    out_dir = Path(args.out_dir) if args.out_dir else resolve_artifact_path(step_record["backtest_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_task_ids = list_seen_task_ids(registry, step_idx=step_idx)
    rows = []
    row_payloads = []
    registry_path = resolve_registry_path(args.cad_root)

    for task_id in seen_task_ids:
        task = registry["tasks"][task_id]
        view_id = int(task["view_id"])
        base_ckpt = resolve_artifact_path(task["base_ckpt"])
        task_dir = out_dir / f"task_{task_id}"

        baseline = run_config(
            args,
            config_name="baseline",
            ckpt=str(base_ckpt),
            view_id=view_id,
            result_dir=task_dir / "baseline",
            depth_peft_ckpt=None,
            cad_registry=None,
        )
        cad_active = run_config(
            args,
            config_name="cad_active",
            ckpt=str(base_ckpt),
            view_id=view_id,
            result_dir=task_dir / "cad_active",
            depth_peft_ckpt=None,
            cad_registry=str(registry_path),
        )
        shared_current = run_config(
            args,
            config_name="shared_current_task",
            ckpt=str(base_ckpt),
            view_id=view_id,
            result_dir=task_dir / "shared_current_task",
            depth_peft_ckpt=str(current_peft_ckpt),
            cad_registry=None,
        )

        rows.append(build_summary_row(task_id, view_id, str(base_ckpt), baseline, cad_active, shared_current))
        row_payloads.append(
            {
                "task_id": str(task_id),
                "view_id": int(view_id),
                "baseline": baseline,
                "cad_active": cad_active,
                "shared_current_task": shared_current,
            }
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_csv = out_dir / "summary.csv"
    summary_txt = out_dir / "summary.txt"
    summary_json = out_dir / "summary.json"
    matrix_json = out_dir / "matrix.json"
    matrix_csv = out_dir / "matrix.csv"
    metrics_json = out_dir / "continual_metrics.json"

    write_summary_csv(summary_csv, rows)

    historical_payloads = []
    for history_step in registry["steps"][:step_idx]:
        history_backtest = resolve_artifact_path(history_step["backtest_dir"]) / "summary.json"
        if history_backtest.exists():
            with open(history_backtest, "r", encoding="utf-8") as f:
                historical_payloads.append(json.load(f))
    current_summary = {
        "step_idx": step_idx,
        "current_task_id": current_task_id,
        "rows": row_payloads,
    }
    all_summaries = historical_payloads + [current_summary]
    matrix_payload = build_matrix_payload(all_summaries, task_order=registry.get("task_order", []), metric_source=args.score_source)
    metrics_payload = compute_continual_metrics(matrix_payload)

    write_matrix_csv(matrix_csv, matrix_payload)
    write_summary_txt(summary_txt, args, registry, step_idx, current_task_id, rows, metrics_payload)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(current_summary, f, ensure_ascii=False, indent=2)
    with open(matrix_json, "w", encoding="utf-8") as f:
        json.dump(matrix_payload, f, ensure_ascii=False, indent=2)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print(f"Summary CSV: {summary_csv}")
    print(f"Summary TXT: {summary_txt}")
    print(f"Matrix CSV: {matrix_csv}")
    print(f"Metrics JSON: {metrics_json}")


if __name__ == "__main__":
    main()
