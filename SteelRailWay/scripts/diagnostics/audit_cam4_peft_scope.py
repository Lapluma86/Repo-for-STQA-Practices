#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit whether Cam4 P1/PEFT training updates only gain/bias and leaves CF/CA untouched."""

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
from rail_peft import DepthAffinePEFT, DepthEncoderWithPEFT, fdm_loss
from scripts.eval.eval_from_ckpt import resolve_amp_dtype, safe_load_ckpt, strip_prefix


DEFAULT_BASELINE_CKPT = (
    "outputs/rail_all/Cam4/20260501_014226_cam4_bs32_lr0.005_img512_ratio1.0/best_cam4.pth"
)
DEFAULT_PEFT_CKPT = "outputs/rail_peft/cam4_p1_20260501_225618/final/final_peft_cam4.pth"
DEFAULT_REFERENCE_STATS = "outputs/rail_peft/cam4_p1_20260501_225618/stats/reference_stats.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Cam4 PEFT trainable scope and verify CF/CA params stay unchanged after one smoke epoch."
    )
    parser.add_argument("--baseline_ckpt", type=str, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--peft_ckpt", type=str, default=DEFAULT_PEFT_CKPT)
    parser.add_argument("--reference_stats", type=str, default=DEFAULT_REFERENCE_STATS)
    parser.add_argument("--test_root", type=str, default="rail_mvtec_gt_test")
    parser.add_argument("--train_root", type=str, default="data_20260327")
    parser.add_argument("--view_id", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--depth_norm", type=str, default="zscore", choices=["zscore", "minmax", "log"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, default="fp32", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--no_channels_last", action="store_false", dest="channels_last")
    parser.add_argument("--out_dir", type=str, default="outputs/rail_ablation/cam4_peft_scope_audit")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_reference_stats(path: Path, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    payload = safe_load_ckpt(path, device)
    if not isinstance(payload, dict) or "mu" not in payload or "var" not in payload:
        raise TypeError(f"Invalid reference_stats payload: {path}")
    mu_ref = [t.to(device=device, dtype=torch.float32) for t in payload["mu"]]
    var_ref = [t.to(device=device, dtype=torch.float32) for t in payload["var"]]
    return mu_ref, var_ref


def load_models(args, device: torch.device):
    teacher_depth = ResNet50Encoder(pretrained=True).to(device).eval()
    teacher_rgb = ResNet50Encoder(pretrained=True).to(device).eval()
    for param in teacher_depth.parameters():
        param.requires_grad_(False)
    for param in teacher_rgb.parameters():
        param.requires_grad_(False)

    student_rgb = ResNet50DualModalDecoder(pretrained=False).to(device).eval()
    student_depth = ResNet50DualModalDecoder(pretrained=False).to(device).eval()
    ckpt = safe_load_ckpt(args.baseline_ckpt, device)
    student_rgb.load_state_dict(strip_prefix(ckpt["student_rgb"]))
    student_depth.load_state_dict(strip_prefix(ckpt["student_depth"]))

    peft_payload = safe_load_ckpt(args.peft_ckpt, device)
    peft_state = peft_payload["state_dict"] if isinstance(peft_payload, dict) and "state_dict" in peft_payload else peft_payload
    peft_state = {
        k.replace("_orig_mod.", "").replace("module.", "").replace("peft.", ""): v
        for k, v in peft_state.items()
        if k.replace("_orig_mod.", "").replace("module.", "").replace("peft.", "") in {"gain", "bias"}
    }
    peft = DepthAffinePEFT().to(device)
    peft.load_state_dict(peft_state, strict=True)
    depth_branch = DepthEncoderWithPEFT(teacher_depth, peft).to(device)

    if args.channels_last and device.type == "cuda":
        teacher_rgb = teacher_rgb.to(memory_format=torch.channels_last)
        teacher_depth = teacher_depth.to(memory_format=torch.channels_last)
        student_rgb = student_rgb.to(memory_format=torch.channels_last)
        student_depth = student_depth.to(memory_format=torch.channels_last)
        depth_branch = depth_branch.to(memory_format=torch.channels_last)

    return teacher_rgb, teacher_depth, student_rgb, student_depth, depth_branch


def collect_named_params(module: torch.nn.Module, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in module.named_parameters()
        if name.startswith(prefixes)
    }


def max_abs_diff(before: dict[str, torch.Tensor], after_module: torch.nn.Module, prefixes: tuple[str, ...]) -> tuple[float, dict[str, float]]:
    deltas = {}
    for name, param in after_module.named_parameters():
        if not name.startswith(prefixes):
            continue
        prev = before[name]
        delta = float((param.detach().cpu() - prev).abs().max().item())
        deltas[name] = delta
    overall = max(deltas.values()) if deltas else 0.0
    return overall, deltas


def build_smoke_loader(args, device: torch.device) -> DataLoader:
    dataset = RailDualModalDataset(
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        split="test",
        img_size=args.img_size,
        depth_norm=args.depth_norm,
        use_patch=True,
        patch_size=900,
        patch_stride=850,
        preload=False,
        preload_workers=0,
    )
    good_indices = []
    for idx, sample in enumerate(dataset.samples):
        if int(sample["label"]) == 0:
            start = idx * dataset.num_patches
            good_indices.extend(range(start, start + dataset.num_patches))
    subset = Subset(dataset, good_indices)
    return DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.precision, device)
    amp_ctx_factory = (
        (lambda: torch.amp.autocast(device.type, enabled=True, dtype=amp_dtype))
        if amp_dtype is not None and hasattr(torch, "amp")
        else nullcontext
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_rgb, teacher_depth, student_rgb, student_depth, depth_branch = load_models(args, device)
    mu_ref, var_ref = load_reference_stats(Path(args.reference_stats), device)

    optimizer = torch.optim.Adam(depth_branch.trainable_parameters(), lr=args.lr)
    optimizer_names = ["peft.gain", "peft.bias"]

    cf_prefixes = ("projector_filter.", "bn.")
    ca_prefixes = ("projector_amply.", "decoder.w_")
    cf_before_rgb = collect_named_params(student_rgb, cf_prefixes)
    cf_before_depth = collect_named_params(student_depth, cf_prefixes)
    ca_before_rgb = collect_named_params(student_rgb, ca_prefixes)
    ca_before_depth = collect_named_params(student_depth, ca_prefixes)

    smoke_loader = build_smoke_loader(args, device)
    loss_values = []
    depth_branch.encoder.eval()
    depth_branch.peft.train()
    for batch_idx, batch in enumerate(smoke_loader):
        depth = batch["depth"].to(device, non_blocking=True)
        if args.channels_last and device.type == "cuda":
            depth = depth.contiguous(memory_format=torch.channels_last)
        with amp_ctx_factory():
            feats = depth_branch(depth)
            loss = fdm_loss(feats, mu_ref, var_ref)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_values.append(float(loss.detach().cpu()))
        if batch_idx >= 0:
            break

    cf_rgb_overall, cf_rgb_deltas = max_abs_diff(cf_before_rgb, student_rgb, cf_prefixes)
    cf_depth_overall, cf_depth_deltas = max_abs_diff(cf_before_depth, student_depth, cf_prefixes)
    ca_rgb_overall, ca_rgb_deltas = max_abs_diff(ca_before_rgb, student_rgb, ca_prefixes)
    ca_depth_overall, ca_depth_deltas = max_abs_diff(ca_before_depth, student_depth, ca_prefixes)

    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_ckpt": args.baseline_ckpt,
        "peft_ckpt": args.peft_ckpt,
        "reference_stats": args.reference_stats,
        "optimizer_param_names": optimizer_names,
        "optimizer_unique_trainable_ok": optimizer_names == ["peft.gain", "peft.bias"],
        "smoke_batches": len(loss_values),
        "smoke_loss_values": loss_values,
        "cf": {
            "rgb_max_abs_diff": cf_rgb_overall,
            "depth_max_abs_diff": cf_depth_overall,
            "rgb_param_deltas": cf_rgb_deltas,
            "depth_param_deltas": cf_depth_deltas,
        },
        "ca": {
            "rgb_max_abs_diff": ca_rgb_overall,
            "depth_max_abs_diff": ca_depth_overall,
            "rgb_param_deltas": ca_rgb_deltas,
            "depth_param_deltas": ca_depth_deltas,
        },
    }

    with open(out_dir / "audit.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(out_dir / "optimizer_params.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name"])
        writer.writeheader()
        for name in optimizer_names:
            writer.writerow({"name": name})

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Cam4 PEFT scope audit\n")
        f.write(f"created_at: {payload['created_at']}\n")
        f.write(f"baseline_ckpt: {args.baseline_ckpt}\n")
        f.write(f"peft_ckpt: {args.peft_ckpt}\n")
        f.write(f"reference_stats: {args.reference_stats}\n")
        f.write(f"optimizer_param_names: {optimizer_names}\n")
        f.write(f"optimizer_unique_trainable_ok: {payload['optimizer_unique_trainable_ok']}\n")
        f.write(f"smoke_batches: {len(loss_values)}\n")
        if loss_values:
            f.write(f"smoke_loss: {loss_values[0]:.8f}\n")
        f.write(
            f"CF max_abs_diff: rgb={cf_rgb_overall:.8e}, depth={cf_depth_overall:.8e}\n"
        )
        f.write(
            f"CA max_abs_diff: rgb={ca_rgb_overall:.8e}, depth={ca_depth_overall:.8e}\n"
        )

    print(f"Audit written to: {out_dir}")


if __name__ == "__main__":
    main()
