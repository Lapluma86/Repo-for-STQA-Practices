#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate Cam4 cf_ca_repair checkpoint on other camera views."""

# >>> path-bootstrap >>>
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
# <<< path-bootstrap <<<

import argparse
import csv
import json
from datetime import datetime
from types import SimpleNamespace

from scripts.eval.eval_from_ckpt import evaluate_from_args


DEFAULT_BASELINE_CKPTS = {
    1: "outputs/rail_all/Cam1/20260430_233006_cam1_bs32_lr0.005_img512_ratio1.0/best_cam1.pth",
    5: "outputs/rail_all/Cam5/20260501_022643_cam5_bs32_lr0.005_img512_ratio1.0/best_cam5.pth",
}
DEFAULT_CAM4_PEFT_CKPT = "outputs/rail_peft/cam4_p1_20260501_225618/final/final_peft_cam4.pth"
DEFAULT_CFCA_REPAIR_CKPT = (
    "outputs/rail_ablation/cam4_cfca_repair/cf_ca/"
    "cam4_cf_ca_20260504_170923/final/repair_cam4.pth"
)


def default_out_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"cam4_cfca_repair_cross_camera_{timestamp}"


def load_ckpt_map(value: str | None) -> dict[int, str]:
    if not value:
        return dict(DEFAULT_BASELINE_CKPTS)
    path = Path(value)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = json.loads(value)
    merged = dict(DEFAULT_BASELINE_CKPTS)
    merged.update({int(k): str(v) for k, v in raw.items() if v})
    return merged


def build_parser():
    parser = argparse.ArgumentParser(
        description="Cross-camera anti-forgetting eval for Cam4 cf_ca_repair on Cam1/Cam5."
    )
    parser.add_argument("--views", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--baseline_ckpt_map", type=str, default=None,
                        help='可选：JSON 文件或 JSON 字符串，如 {"1": "path/to/best_cam1.pth"}')
    parser.add_argument("--cam4_peft_ckpt", type=str, default=DEFAULT_CAM4_PEFT_CKPT)
    parser.add_argument("--cfca_repair_ckpt", type=str, default=DEFAULT_CFCA_REPAIR_CKPT)
    parser.add_argument("--train_root", type=str, default="G:/SteelRailWay/data_20260327")
    parser.add_argument("--test_root", type=str, default="G:/SteelRailWay/rail_mvtec_gt_test")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--depth_norm", type=str, default="zscore",
                        choices=["zscore", "minmax", "log"])
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--use_patch", action="store_true", default=True)
    parser.add_argument("--no_patch", action="store_false", dest="use_patch")
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--preload", action="store_true", default=False)
    parser.add_argument("--preload_workers", type=int, default=8)
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--no_channels_last", action="store_false", dest="channels_last")
    parser.add_argument("--assist_fill", type=str, default="zeros",
                        choices=["train_mean", "zeros"])
    parser.add_argument("--assist_stats_dir", type=str, default=None)
    parser.add_argument("--train_sample_ratio", type=float, default=1.0)
    parser.add_argument("--train_sample_num", type=int, default=1200)
    parser.add_argument("--sampling_mode", type=str, default="uniform_time",
                        choices=["uniform_time", "random"])
    parser.add_argument("--train_sample_seed", type=int, default=42)
    parser.add_argument("--fusion_rule", type=str, default="sum",
                        choices=["sum", "max_norm"])
    return parser


def make_eval_args(args, ckpt: str, view_id: int, config_dir: Path, depth_peft_ckpt: str | None):
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
        result_json=str(config_dir / "result.json"),
        depth_peft_ckpt=str(depth_peft_ckpt or ""),
        module_ablation="full",
        score_source="fusion",
        fusion_rule=args.fusion_rule,
        scores_dir=str(config_dir),
        assist_fill=args.assist_fill,
        assist_stats_dir=str(args.assist_stats_dir or ""),
        train_sample_ratio=args.train_sample_ratio,
        train_sample_num=args.train_sample_num,
        sampling_mode=args.sampling_mode,
        train_sample_seed=args.train_sample_seed,
    )


def fmt(value):
    return "" if value is None else f"{float(value):.8f}"


def metric_or_none(result: dict, source: str):
    return result.get("auroc_by_source", {}).get(source)


def result_path_for(out_dir: Path, view_id: int, config_name: str) -> Path:
    return out_dir / f"cam{int(view_id)}_{config_name}"


def build_row(view_id: int, config_name: str, ckpt: str, depth_peft_ckpt: str | None, result: dict) -> dict:
    return {
        "view_id": str(view_id),
        "config_name": config_name,
        "status": str(result.get("status", "")),
        "reason": str(result.get("reason", "")),
        "ckpt": str(ckpt),
        "depth_peft_ckpt": str(depth_peft_ckpt or ""),
        "num_images": str(result.get("num_images", "")),
        "num_abnormal": str(result.get("num_abnormal", "")),
        "num_normal": str(result.get("num_normal", "")),
        "auroc_rgb": fmt(metric_or_none(result, "rgb")),
        "auroc_depth": fmt(metric_or_none(result, "depth")),
        "auroc_fusion": fmt(metric_or_none(result, "fusion")),
        "auroc_rgb_isolated": fmt(metric_or_none(result, "rgb_isolated")),
        "auroc_depth_isolated": fmt(metric_or_none(result, "depth_isolated")),
        "delta_rgb_vs_baseline": "",
        "delta_depth_vs_baseline": "",
        "delta_fusion_vs_baseline": "",
        "delta_rgb_isolated_vs_baseline": "",
        "delta_depth_isolated_vs_baseline": "",
        "delta_rgb_vs_cam4peft": "",
        "delta_depth_vs_cam4peft": "",
        "delta_fusion_vs_cam4peft": "",
        "delta_rgb_isolated_vs_cam4peft": "",
        "delta_depth_isolated_vs_cam4peft": "",
        "result_json": str(result.get("result_json", "")),
    }


def set_delta(row: dict, key: str, current: str, reference: str):
    if current == "" or reference == "":
        row[key] = ""
        return
    row[key] = f"{float(current) - float(reference):.8f}"


def attach_deltas(rows: list[dict]) -> None:
    refs = {}
    for row in rows:
        refs[(int(row["view_id"]), row["config_name"])] = row

    metric_keys = [
        ("auroc_rgb", "delta_rgb_vs_baseline", "delta_rgb_vs_cam4peft"),
        ("auroc_depth", "delta_depth_vs_baseline", "delta_depth_vs_cam4peft"),
        ("auroc_fusion", "delta_fusion_vs_baseline", "delta_fusion_vs_cam4peft"),
        ("auroc_rgb_isolated", "delta_rgb_isolated_vs_baseline", "delta_rgb_isolated_vs_cam4peft"),
        ("auroc_depth_isolated", "delta_depth_isolated_vs_baseline", "delta_depth_isolated_vs_cam4peft"),
    ]
    for row in rows:
        view_id = int(row["view_id"])
        baseline = refs.get((view_id, "baseline"))
        peft = refs.get((view_id, "with_cam4peft"))
        for metric_key, baseline_key, peft_key in metric_keys:
            if baseline:
                set_delta(row, baseline_key, row.get(metric_key, ""), baseline.get(metric_key, ""))
            if peft:
                set_delta(row, peft_key, row.get(metric_key, ""), peft.get(metric_key, ""))


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "view_id",
        "config_name",
        "status",
        "reason",
        "ckpt",
        "depth_peft_ckpt",
        "num_images",
        "num_abnormal",
        "num_normal",
        "auroc_rgb",
        "auroc_depth",
        "auroc_fusion",
        "auroc_rgb_isolated",
        "auroc_depth_isolated",
        "delta_rgb_vs_baseline",
        "delta_depth_vs_baseline",
        "delta_fusion_vs_baseline",
        "delta_rgb_isolated_vs_baseline",
        "delta_depth_isolated_vs_baseline",
        "delta_rgb_vs_cam4peft",
        "delta_depth_vs_cam4peft",
        "delta_fusion_vs_cam4peft",
        "delta_rgb_isolated_vs_cam4peft",
        "delta_depth_isolated_vs_cam4peft",
        "result_json",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_row(rows: list[dict], view_id: int, config_name: str) -> dict | None:
    for row in rows:
        if int(row["view_id"]) == int(view_id) and row["config_name"] == config_name:
            return row
    return None


def write_summary_txt(path: Path, args, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Cam4 cf_ca_repair cross-camera evaluation\n")
        f.write(f"created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"views: {' '.join(map(str, args.views))}\n")
        f.write(f"train_root: {args.train_root}\n")
        f.write(f"test_root: {args.test_root}\n")
        f.write(f"cam4_peft_ckpt: {args.cam4_peft_ckpt}\n")
        f.write(f"cfca_repair_ckpt: {args.cfca_repair_ckpt}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"precision: {args.precision}\n")
        f.write(f"assist_fill: {args.assist_fill}\n")
        f.write(f"fusion_rule: {args.fusion_rule}\n\n")

        for view_id in args.views:
            baseline = find_row(rows, view_id, "baseline")
            peft = find_row(rows, view_id, "with_cam4peft")
            repair = find_row(rows, view_id, "with_cam4_cfca_repair")
            if not baseline or not peft or not repair:
                continue
            f.write(f"Cam{view_id}\n")
            f.write(
                f"  baseline: rgb={baseline['auroc_rgb']}, depth={baseline['auroc_depth']}, "
                f"fusion={baseline['auroc_fusion']}\n"
            )
            f.write(
                f"  with_cam4peft: rgb={peft['auroc_rgb']}, depth={peft['auroc_depth']}, "
                f"fusion={peft['auroc_fusion']}, d_fusion_vs_baseline={peft['delta_fusion_vs_baseline']}\n"
            )
            f.write(
                f"  with_cam4_cfca_repair: rgb={repair['auroc_rgb']}, depth={repair['auroc_depth']}, "
                f"fusion={repair['auroc_fusion']}, d_fusion_vs_baseline={repair['delta_fusion_vs_baseline']}, "
                f"d_fusion_vs_cam4peft={repair['delta_fusion_vs_cam4peft']}\n"
            )
            f.write(
                f"  isolated: rgb={repair['auroc_rgb_isolated']}, depth={repair['auroc_depth_isolated']}, "
                f"d_rgb_iso_vs_cam4peft={repair['delta_rgb_isolated_vs_cam4peft']}, "
                f"d_depth_iso_vs_cam4peft={repair['delta_depth_isolated_vs_cam4peft']}\n\n"
            )


def write_summary_json(path: Path, args, rows: list[dict]) -> None:
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "views": list(args.views),
        "train_root": args.train_root,
        "test_root": args.test_root,
        "cam4_peft_ckpt": args.cam4_peft_ckpt,
        "cfca_repair_ckpt": args.cfca_repair_ckpt,
        "assist_fill": args.assist_fill,
        "fusion_rule": args.fusion_rule,
        "rows": rows,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_ckpts = load_ckpt_map(args.baseline_ckpt_map)

    configs = [
        ("baseline", None, "baseline"),
        ("with_cam4peft", None, "peft"),
        ("with_cam4_cfca_repair", args.cfca_repair_ckpt, "repair"),
    ]

    rows = []
    for view_id in args.views:
        if view_id not in baseline_ckpts:
            raise KeyError(f"Missing baseline ckpt for Cam{view_id}")
        baseline_ckpt = baseline_ckpts[view_id]
        for config_name, ckpt_override, depth_mode in configs:
            config_dir = result_path_for(out_dir, view_id, config_name)
            config_dir.mkdir(parents=True, exist_ok=True)
            ckpt = ckpt_override or baseline_ckpt
            depth_peft_ckpt = args.cam4_peft_ckpt if depth_mode != "baseline" else None
            eval_args = make_eval_args(args, ckpt, view_id, config_dir, depth_peft_ckpt)
            print("=" * 72)
            print(f"Evaluating Cam{view_id} / {config_name}")
            print(f"  ckpt={ckpt}")
            print(f"  depth_peft_ckpt={depth_peft_ckpt or 'N/A'}")
            result = evaluate_from_args(eval_args)
            result["result_json"] = str(config_dir / "result.json")
            rows.append(build_row(view_id, config_name, ckpt, depth_peft_ckpt, result))

    attach_deltas(rows)

    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"
    summary_txt = out_dir / "summary.txt"
    write_summary_csv(summary_csv, rows)
    write_summary_json(summary_json, args, rows)
    write_summary_txt(summary_txt, args, rows)

    print("=" * 72)
    print("Done")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary TXT: {summary_txt}")


if __name__ == "__main__":
    main()
