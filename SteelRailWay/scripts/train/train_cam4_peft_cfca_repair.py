#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Repair-train Cam4 CF/CA modules after PEFT using normal-only teacher-student feature alignment."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from datasets.rail_dataset import RailDualModalDataset
from models.trd.decoder import ResNet50DualModalDecoder
from models.trd.encoder import ResNet50Encoder
from rail_peft import DepthEncoderWithPEFT
from scripts.eval.eval_from_ckpt import evaluate_from_args, load_depth_peft, resolve_amp_dtype, safe_load_ckpt, strip_prefix
from utils.losses import loss_distil


DEFAULT_BASELINE_CKPT = (
    "outputs/rail_all/Cam4/20260501_014226_cam4_bs32_lr0.005_img512_ratio1.0/best_cam4.pth"
)
DEFAULT_PEFT_CKPT = "outputs/rail_peft/cam4_p1_20260501_225618/final/final_peft_cam4.pth"
SCOPES = ("cf_only", "ca_only", "cf_ca")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train small-scope CF/CA repair adapters on Cam4 good samples after PEFT."
    )
    parser.add_argument("--ckpt", type=str, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--depth_peft_ckpt", type=str, default=DEFAULT_PEFT_CKPT)
    parser.add_argument("--train_root", type=str, default="data_20260327")
    parser.add_argument("--test_root", type=str, default="rail_mvtec_gt_test")
    parser.add_argument("--view_id", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--depth_norm", type=str, default="zscore", choices=["zscore", "minmax", "log"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, default="fp32", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--no_channels_last", action="store_false", dest="channels_last")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--scope", type=str, nargs="+", default=list(SCOPES), choices=SCOPES)
    parser.add_argument("--output_root", type=str, default="outputs/rail_ablation/cam4_cfca_repair")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader_kwargs(num_workers: int, device: torch.device):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def build_test_dataset(args) -> RailDualModalDataset:
    return RailDualModalDataset(
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        split="test",
        img_size=args.img_size,
        depth_norm=args.depth_norm,
        use_patch=True,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        preload=False,
        preload_workers=0,
    )


def indices_for_frames(dataset: RailDualModalDataset, frame_ids: set[str]) -> list[int]:
    indices = []
    for img_idx, sample in enumerate(dataset.samples):
        if sample["frame_id"] in frame_ids:
            start = img_idx * dataset.num_patches
            indices.extend(range(start, start + dataset.num_patches))
    return indices


def make_loader(dataset, indices, batch_size, num_workers, device, shuffle=False):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        **loader_kwargs(num_workers, device),
    )


def build_models(args, device: torch.device, baseline_ckpt_cpu):
    teacher_rgb = ResNet50Encoder(pretrained=True).to(device).eval()
    teacher_depth_base = ResNet50Encoder(pretrained=True).to(device).eval()
    for p in teacher_rgb.parameters():
        p.requires_grad_(False)
    for p in teacher_depth_base.parameters():
        p.requires_grad_(False)

    peft, peft_meta = load_depth_peft(args.depth_peft_ckpt, torch.device("cpu"))
    for p in peft.parameters():
        p.requires_grad_(False)
    teacher_depth = DepthEncoderWithPEFT(teacher_depth_base, peft).to(device).eval()

    student_rgb = ResNet50DualModalDecoder(pretrained=False).to(device)
    student_depth = ResNet50DualModalDecoder(pretrained=False).to(device)
    student_rgb.load_state_dict(strip_prefix(baseline_ckpt_cpu["student_rgb"]))
    student_depth.load_state_dict(strip_prefix(baseline_ckpt_cpu["student_depth"]))

    if args.channels_last and device.type == "cuda":
        teacher_rgb = teacher_rgb.to(memory_format=torch.channels_last)
        teacher_depth = teacher_depth.to(memory_format=torch.channels_last)
        student_rgb = student_rgb.to(memory_format=torch.channels_last)
        student_depth = student_depth.to(memory_format=torch.channels_last)

    return teacher_rgb, teacher_depth, student_rgb, student_depth, baseline_ckpt_cpu, peft_meta


def freeze_all(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def unfreeze_scope(student: ResNet50DualModalDecoder, scope: str) -> list[str]:
    prefixes = []
    if scope in {"cf_only", "cf_ca"}:
        prefixes.extend(["projector_filter.", "bn."])
    if scope in {"ca_only", "cf_ca"}:
        prefixes.extend(["projector_amply.", "decoder.w_"])
    trainable = []
    for name, param in student.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad_(True)
            trainable.append(name)
    return trainable


def train_one_epoch(
    teacher_rgb,
    teacher_depth,
    student_rgb,
    student_depth,
    loader,
    optimizer,
    device,
    amp_dtype,
    channels_last,
) -> float:
    student_rgb.train()
    student_depth.train()
    loss_values = []
    amp_ctx = (
        (lambda: torch.amp.autocast(device.type, enabled=True, dtype=amp_dtype))
        if amp_dtype is not None and hasattr(torch, "amp")
        else nullcontext
    )

    for batch in loader:
        rgb = batch["intensity"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            rgb = rgb.contiguous(memory_format=torch.channels_last)
            depth = depth.contiguous(memory_format=torch.channels_last)

        with torch.no_grad():
            with amp_ctx():
                feat_t_rgb = teacher_rgb(rgb)
                feat_t_depth = teacher_depth(depth)

        with amp_ctx():
            _, _, feat_s_rgb, _ = student_rgb(feat_t_rgb, feat_t_depth)
            _, _, feat_s_depth, _ = student_depth(feat_t_depth, feat_t_rgb)
            loss = loss_distil(feat_s_rgb, feat_t_rgb) + loss_distil(feat_s_depth, feat_t_depth)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_values.append(float(loss.detach().cpu()))

    return float(np.mean(loss_values)) if loss_values else 0.0


def evaluate_repair(args, ckpt_path: Path, out_dir: Path) -> dict:
    eval_args = argparse.Namespace(
        ckpt=str(ckpt_path),
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        img_size=args.img_size,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=args.device,
        depth_norm=args.depth_norm,
        use_patch=True,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        preload=False,
        preload_workers=0,
        precision=args.precision,
        channels_last=args.channels_last,
        output_log=None,
        append_log=False,
        scores_csv=None,
        result_json=str(out_dir / "result.json"),
        depth_peft_ckpt=args.depth_peft_ckpt,
        module_ablation="full",
        score_source="fusion",
        fusion_rule="sum",
        scores_dir=str(out_dir),
        assist_fill="zeros",
        assist_stats_dir=None,
        train_sample_ratio=1.0,
        train_sample_num=None,
        sampling_mode="uniform_time",
        train_sample_seed=42,
    )
    return evaluate_from_args(eval_args)


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.precision, device)

    log("Building Cam4 test dataset for repair training")
    dataset = build_test_dataset(args)
    good_ids = sorted([s["frame_id"] for s in dataset.samples if int(s["label"]) == 0])
    broken_ids = sorted([s["frame_id"] for s in dataset.samples if int(s["label"]) == 1])
    if len(good_ids) < args.folds:
        raise RuntimeError(f"Need at least {args.folds} good images, got {len(good_ids)}")

    fold_parts = [list(part) for part in np.array_split(np.array(good_ids), args.folds)]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    log(
        f"Dataset ready: good={len(good_ids)}, broken={len(broken_ids)}, "
        f"patches_per_image={dataset.num_patches}, scopes={list(args.scope)}, epochs={args.epochs}, folds={args.folds}"
    )
    log("Loading baseline checkpoint once on CPU (large file; first time may take a while)")
    baseline_ckpt_cpu = safe_load_ckpt(args.ckpt, torch.device("cpu"))
    log("Baseline checkpoint loaded")
    log("First teacher build may download ImageNet weights to ~/.cache/torch/hub/checkpoints on cache miss")

    for scope in args.scope:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scope_root = output_root / scope / f"cam4_{scope}_{timestamp}"
        scope_root.mkdir(parents=True, exist_ok=True)
        log(f"Starting scope={scope} -> {scope_root}")

        cv_rows = []
        for fold_idx, test_good in enumerate(fold_parts, start=1):
            log(f"[{scope}] Building models for fold {fold_idx}/{args.folds}")
            teacher_rgb, teacher_depth, student_rgb, student_depth, ckpt_meta, peft_meta = build_models(
                args, device, baseline_ckpt_cpu
            )
            freeze_all(student_rgb)
            freeze_all(student_depth)
            trainable_rgb = unfreeze_scope(student_rgb, scope)
            trainable_depth = unfreeze_scope(student_depth, scope)
            params = [p for p in student_rgb.parameters() if p.requires_grad] + [p for p in student_depth.parameters() if p.requires_grad]
            if not params:
                raise RuntimeError(f"No trainable parameters found for scope={scope}")
            optimizer = torch.optim.Adam(params, lr=args.lr)

            train_good = [fid for fid in good_ids if fid not in set(test_good)]
            train_indices = indices_for_frames(dataset, set(train_good))
            train_loader = make_loader(dataset, train_indices, args.batch_size, args.num_workers, device, shuffle=True)
            log(
                f"[{scope}] Fold {fold_idx}: train_good={len(train_good)}, heldout_good={len(test_good)}, "
                f"train_patches={len(train_indices)}, batch_size={args.batch_size}"
            )

            history = []
            for epoch in range(1, args.epochs + 1):
                avg_loss = train_one_epoch(
                    teacher_rgb, teacher_depth, student_rgb, student_depth,
                    train_loader, optimizer, device, amp_dtype, args.channels_last,
                )
                if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
                    history.append({"epoch": epoch, "loss": f"{avg_loss:.8f}"})
                    log(f"[{scope}] Fold {fold_idx}: epoch {epoch}/{args.epochs}, loss={avg_loss:.8f}")

            fold_dir = scope_root / "cv" / f"fold{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_ckpt = fold_dir / f"repair_cam{args.view_id}.pth"
            torch.save({
                "epoch": args.epochs,
                "student_rgb": student_rgb.state_dict(),
                "student_depth": student_depth.state_dict(),
                "best_val_loss": float(history[-1]["loss"]) if history else None,
            }, fold_ckpt)
            with open(fold_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "loss"])
                writer.writeheader()
                writer.writerows(history)
            log(f"[{scope}] Fold {fold_idx}: training done, starting evaluation")
            eval_result = evaluate_repair(args, fold_ckpt, fold_dir / "eval")
            log(
                f"[{scope}] Fold {fold_idx}: eval done, "
                f"fusion={float(eval_result['auroc_by_source']['fusion']):.8f}"
            )
            cv_rows.append({
                "scope": scope,
                "fold": fold_idx,
                "rgb_auroc": f"{float(eval_result['auroc_by_source']['rgb']):.8f}",
                "depth_auroc": f"{float(eval_result['auroc_by_source']['depth']):.8f}",
                "fusion_auroc": f"{float(eval_result['auroc_by_source']['fusion']):.8f}",
                "train_good": " ".join(train_good),
                "test_good": " ".join(test_good),
                "ckpt": str(fold_ckpt),
            })

        log(f"[{scope}] Building models for final train on all good samples")
        teacher_rgb, teacher_depth, student_rgb, student_depth, ckpt_meta, peft_meta = build_models(
            args, device, baseline_ckpt_cpu
        )
        freeze_all(student_rgb)
        freeze_all(student_depth)
        trainable_rgb = unfreeze_scope(student_rgb, scope)
        trainable_depth = unfreeze_scope(student_depth, scope)
        params = [p for p in student_rgb.parameters() if p.requires_grad] + [p for p in student_depth.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(f"No trainable parameters found for scope={scope}")
        optimizer = torch.optim.Adam(params, lr=args.lr)

        final_train_indices = indices_for_frames(dataset, set(good_ids))
        final_loader = make_loader(dataset, final_train_indices, args.batch_size, args.num_workers, device, shuffle=True)
        log(f"[{scope}] Final train: good_images={len(good_ids)}, train_patches={len(final_train_indices)}")
        final_history = []
        for epoch in range(1, args.epochs + 1):
            avg_loss = train_one_epoch(
                teacher_rgb, teacher_depth, student_rgb, student_depth,
                final_loader, optimizer, device, amp_dtype, args.channels_last,
            )
            if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
                final_history.append({"epoch": epoch, "loss": f"{avg_loss:.8f}"})
                log(f"[{scope}] Final: epoch {epoch}/{args.epochs}, loss={avg_loss:.8f}")

        final_dir = scope_root / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_ckpt = final_dir / f"repair_cam{args.view_id}.pth"
        torch.save({
            "epoch": args.epochs,
            "student_rgb": student_rgb.state_dict(),
            "student_depth": student_depth.state_dict(),
            "best_val_loss": float(final_history[-1]["loss"]) if final_history else None,
        }, final_ckpt)
        with open(final_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "loss"])
            writer.writeheader()
            writer.writerows(final_history)

        log(f"[{scope}] Final training done, starting final evaluation")
        final_eval = evaluate_repair(args, final_ckpt, final_dir / "eval")
        log(
            f"[{scope}] Final eval done: rgb={float(final_eval['auroc_by_source']['rgb']):.8f}, "
            f"depth={float(final_eval['auroc_by_source']['depth']):.8f}, "
            f"fusion={float(final_eval['auroc_by_source']['fusion']):.8f}"
        )
        summary_rows.append({
            "scheme": scope,
            "rgb_auroc": f"{float(final_eval['auroc_by_source']['rgb']):.8f}",
            "depth_auroc": f"{float(final_eval['auroc_by_source']['depth']):.8f}",
            "fusion_auroc": f"{float(final_eval['auroc_by_source']['fusion']):.8f}",
            "delta_vs_peft_full": f"{float(final_eval['auroc_by_source']['fusion']) - 0.6875:.8f}",
            "trainable_rgb_count": len(trainable_rgb),
            "trainable_depth_count": len(trainable_depth),
            "final_ckpt": str(final_ckpt),
        })

        with open(scope_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(cv_rows[0].keys()) if cv_rows else [
                "scope", "fold", "rgb_auroc", "depth_auroc", "fusion_auroc", "train_good", "test_good", "ckpt",
            ])
            writer.writeheader()
            writer.writerows(cv_rows)
        with open(scope_root / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "scope": scope,
                "baseline_ckpt": args.ckpt,
                "depth_peft_ckpt": args.depth_peft_ckpt,
                "trainable_rgb": trainable_rgb,
                "trainable_depth": trainable_depth,
                "good_ids": good_ids,
                "broken_ids": broken_ids,
                "ckpt_meta": {
                    "epoch": ckpt_meta.get("epoch", None) if isinstance(ckpt_meta, dict) else None,
                    "best_val_loss": ckpt_meta.get("best_val_loss", None) if isinstance(ckpt_meta, dict) else None,
                },
                "peft_meta": peft_meta,
            }, f, ensure_ascii=False, indent=2)
        log(f"[{scope}] Scope completed")

    out_summary = output_root / "summary.csv"
    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [
            "scheme", "rgb_auroc", "depth_auroc", "fusion_auroc", "delta_vs_peft_full", "trainable_rgb_count",
            "trainable_depth_count", "final_ckpt",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    log(f"Repair summary CSV: {out_summary}")


if __name__ == "__main__":
    main()
