#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train Cam4 in two stages: P1 DepthAffinePEFT, then joint PEFT+CF/CA continuation."""

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
from eval.eval_utils import cal_anomaly_map
from models.trd.decoder import ResNet50DualModalDecoder
from models.trd.encoder import ResNet50Encoder
from rail_peft import DepthAffinePEFT, DepthEncoderWithPEFT, compute_depth_feature_stats, fdm_loss
from scripts.eval.eval_from_ckpt import evaluate_from_args, resolve_amp_dtype, safe_load_ckpt, strip_prefix, autocast
from utils.losses import loss_distil


DEFAULT_BASELINE_CKPT = (
    "outputs/rail_all/Cam4/20260501_014226_cam4_bs32_lr0.005_img512_ratio1.0/best_cam4.pth"
)
DEFAULT_EXISTING_P1 = "outputs/rail_peft/cam4_p1_20260501_225618/final/final_peft_cam4.pth"
DEFAULT_CFCA_REPAIR_SUMMARY = "outputs/rail_ablation/cam4_cfca_repair/summary.csv"
PEFT_FULL_FUSION = 0.6875


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cam4 chain experiment: P1 DepthAffinePEFT followed by joint PEFT+CF/CA continuation."
    )
    parser.add_argument("--ckpt", type=str, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--existing_p1_ckpt", type=str, default=DEFAULT_EXISTING_P1)
    parser.add_argument("--reuse_existing_p1", action="store_true", default=True)
    parser.add_argument("--retrain_p1", action="store_false", dest="reuse_existing_p1")
    parser.add_argument("--train_root", type=str, default="data_20260327")
    parser.add_argument("--test_root", type=str, default="rail_mvtec_gt_test")
    parser.add_argument("--view_id", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--depth_norm", type=str, default="zscore", choices=["zscore", "minmax", "log"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, default="fp32", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--no_channels_last", action="store_false", dest="channels_last")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--p1_epochs", type=int, default=100)
    parser.add_argument("--p1_lr", type=float, default=1e-2)
    parser.add_argument("--p1_batch_size", type=int, default=16)
    parser.add_argument("--p1_stats_batch_size", type=int, default=32)
    parser.add_argument("--stage2_epochs", type=int, default=100)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--stage2_batch_size", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--output_root", type=str, default="outputs/rail_ablation/cam4_peft_full_then_cf_ca")
    parser.add_argument("--cfca_repair_summary", type=str, default=DEFAULT_CFCA_REPAIR_SUMMARY)
    return parser


def amp_context_factory(device: torch.device, amp_dtype):
    if amp_dtype is None or autocast is None:
        return nullcontext
    return lambda: autocast(device.type, enabled=True, dtype=amp_dtype)


def loader_kwargs(num_workers: int, device: torch.device):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def build_run_layout(output_root: Path) -> dict[str, Path]:
    layout = {
        "root": output_root,
        "p1_root": output_root / "p1",
        "stage2_root": output_root / "peft_full_then_cf_ca",
        "summary_csv": output_root / "summary.csv",
        "summary_json": output_root / "summary.json",
        "comparison_csv": output_root / "comparison_summary.csv",
        "metadata_json": output_root / "metadata.json",
    }
    for key in ["root", "p1_root", "stage2_root"]:
        layout[key].mkdir(parents=True, exist_ok=True)
    return layout


def stage_layout(root: Path, ckpt_name: str) -> dict[str, Path]:
    layout = {
        "root": root,
        "cv_root": root / "cv",
        "final_root": root / "final",
        "summary_csv": root / "summary.csv",
        "summary_json": root / "summary.json",
        "metadata_json": root / "metadata.json",
        "final_ckpt": root / "final" / ckpt_name,
        "final_history": root / "final" / "history.csv",
    }
    layout["cv_root"].mkdir(parents=True, exist_ok=True)
    layout["final_root"].mkdir(parents=True, exist_ok=True)
    return layout


def build_stats_dataset(args) -> RailDualModalDataset:
    return RailDualModalDataset(
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        split="train",
        img_size=args.img_size,
        depth_norm=args.depth_norm,
        use_patch=False,
        train_val_test_split=[1.0, 0.0, 0.0],
        sampling_mode="uniform_time",
        preload=False,
        preload_workers=0,
    )


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


def build_p1_models(args, device: torch.device):
    teacher_rgb = ResNet50Encoder(pretrained=True).to(device).eval()
    teacher_depth = ResNet50Encoder(pretrained=True).to(device).eval()
    for p in teacher_rgb.parameters():
        p.requires_grad_(False)
    for p in teacher_depth.parameters():
        p.requires_grad_(False)

    student_rgb = ResNet50DualModalDecoder(pretrained=False).to(device).eval()
    student_depth = ResNet50DualModalDecoder(pretrained=False).to(device).eval()
    ckpt = safe_load_ckpt(args.ckpt, device)
    student_rgb.load_state_dict(strip_prefix(ckpt["student_rgb"]))
    student_depth.load_state_dict(strip_prefix(ckpt["student_depth"]))

    if args.channels_last and device.type == "cuda":
        teacher_rgb = teacher_rgb.to(memory_format=torch.channels_last)
        teacher_depth = teacher_depth.to(memory_format=torch.channels_last)
        student_rgb = student_rgb.to(memory_format=torch.channels_last)
        student_depth = student_depth.to(memory_format=torch.channels_last)
    return teacher_rgb, teacher_depth, student_rgb, student_depth, ckpt


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_history_csv(path: Path, history: list[dict], fieldnames: list[str]) -> None:
    write_csv(path, history, fieldnames)


def save_peft(path: Path, peft: DepthAffinePEFT, args, train_good, test_good=None) -> None:
    payload = {
        "kind": "DepthAffinePEFT",
        "state_dict": peft.state_dict(),
        "view_id": int(args.view_id),
        "gain": float(peft.gain.detach().cpu()),
        "bias": float(peft.bias.detach().cpu()),
        "train_good": list(train_good),
        "test_good": list(test_good or []),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(payload, path)


@torch.no_grad()
def evaluate_p1_detailed(
    teacher_rgb,
    teacher_depth,
    student_rgb,
    student_depth,
    dataloader,
    device: torch.device,
    amp_ctx_factory,
    channels_last: bool,
):
    teacher_rgb.eval()
    teacher_depth.eval()
    student_rgb.eval()
    student_depth.eval()

    img_scores: dict[str, list[float]] = {}
    img_labels: dict[str, int] = {}

    for data in dataloader:
        rgb = data["intensity"].to(device, non_blocking=True)
        depth = data["depth"].to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            rgb = rgb.contiguous(memory_format=torch.channels_last)
            depth = depth.contiguous(memory_format=torch.channels_last)

        labels = data["label"].cpu().numpy()
        frame_ids = data["frame_id"]

        with amp_ctx_factory():
            feat_t_rgb = teacher_rgb(rgb)
            feat_t_depth = teacher_depth(depth)
            _, _, feat_s_rgb, _ = student_rgb(feat_t_rgb, feat_t_depth)
            _, _, feat_s_depth, _ = student_depth(feat_t_depth, feat_t_rgb)

        amap_rgb, _ = cal_anomaly_map(feat_s_rgb, feat_t_rgb, out_size=(256, 256), amap_mode="mul")
        amap_depth, _ = cal_anomaly_map(feat_s_depth, feat_t_depth, out_size=(256, 256), amap_mode="mul")
        amap = amap_rgb + amap_depth

        for idx, frame_id in enumerate(frame_ids):
            img_scores.setdefault(frame_id, []).append(float(np.max(amap[idx])))
            img_labels[frame_id] = int(labels[idx])

    frame_ids = sorted(img_scores.keys())
    scores = np.array([max(img_scores[fid]) for fid in frame_ids], dtype=np.float32)
    labels = np.array([img_labels[fid] for fid in frame_ids], dtype=np.int64)
    auroc = float("nan")
    if len(np.unique(labels)) > 1:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(labels, scores))
    return auroc, frame_ids, scores, labels


def write_scores_csv(path: Path, frame_ids, scores, labels) -> None:
    order = np.argsort(-scores)
    rows = []
    for rank, idx in enumerate(order):
        rows.append(
            {
                "rank": rank,
                "frame_id": frame_ids[idx],
                "score": f"{float(scores[idx]):.8f}",
                "label": int(labels[idx]),
            }
        )
    write_csv(path, rows, ["rank", "frame_id", "score", "label"])


def train_p1(
    teacher_depth,
    train_loader,
    mu_ref,
    var_ref,
    epochs: int,
    lr: float,
    log_every: int,
    device: torch.device,
    channels_last: bool,
    amp_ctx_factory,
) -> tuple[DepthAffinePEFT, list[dict]]:
    peft = DepthAffinePEFT().to(device)
    depth_branch = DepthEncoderWithPEFT(teacher_depth, peft).to(device)
    optimizer = torch.optim.Adam(depth_branch.trainable_parameters(), lr=lr)
    history = []
    mu_ref = [mu.to(device) for mu in mu_ref]
    var_ref = [var.to(device) for var in var_ref]

    for epoch in range(1, epochs + 1):
        depth_branch.encoder.eval()
        depth_branch.peft.train()
        loss_sum = 0.0
        batch_count = 0

        for data in train_loader:
            depth = data["depth"].to(device, non_blocking=True)
            if channels_last and device.type == "cuda":
                depth = depth.contiguous(memory_format=torch.channels_last)

            with amp_ctx_factory():
                feats = depth_branch(depth)
                loss = fdm_loss(feats, mu_ref, var_ref)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batch_count += 1

        avg_loss = loss_sum / max(batch_count, 1)
        row = {
            "epoch": epoch,
            "loss": f"{avg_loss:.8f}",
            "gain": f"{peft.gain.item():.8f}",
            "bias": f"{peft.bias.item():.8f}",
        }
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            history.append(row)
            log(
                f"[P1] epoch={epoch:03d} loss={avg_loss:.8f} "
                f"gain={peft.gain.item():.8f} bias={peft.bias.item():.8f}"
            )
    return peft, history


def save_joint_ckpt(
    path: Path,
    student_rgb: ResNet50DualModalDecoder,
    student_depth: ResNet50DualModalDecoder,
    peft: DepthAffinePEFT,
    epoch: int,
    best_val_loss: float | None,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "student_rgb": student_rgb.state_dict(),
            "student_depth": student_depth.state_dict(),
            "depth_peft": peft.state_dict(),
            "depth_peft_gain": float(peft.gain.detach().cpu()),
            "depth_peft_bias": float(peft.bias.detach().cpu()),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def freeze_all(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def unfreeze_cfca(student: ResNet50DualModalDecoder) -> list[str]:
    prefixes = ["projector_filter.", "bn.", "projector_amply.", "decoder.w_"]
    trainable = []
    for name, param in student.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad_(True)
            trainable.append(name)
    return trainable


def build_stage2_models(args, device: torch.device, baseline_ckpt_cpu, peft_state_dict: dict):
    teacher_rgb = ResNet50Encoder(pretrained=True).to(device).eval()
    teacher_depth_base = ResNet50Encoder(pretrained=True).to(device).eval()
    for p in teacher_rgb.parameters():
        p.requires_grad_(False)
    for p in teacher_depth_base.parameters():
        p.requires_grad_(False)

    peft = DepthAffinePEFT().to(device)
    peft.load_state_dict(peft_state_dict, strict=True)
    teacher_depth = DepthEncoderWithPEFT(teacher_depth_base, peft).to(device)

    student_rgb = ResNet50DualModalDecoder(pretrained=False).to(device)
    student_depth = ResNet50DualModalDecoder(pretrained=False).to(device)
    student_rgb.load_state_dict(strip_prefix(baseline_ckpt_cpu["student_rgb"]))
    student_depth.load_state_dict(strip_prefix(baseline_ckpt_cpu["student_depth"]))

    if args.channels_last and device.type == "cuda":
        teacher_rgb = teacher_rgb.to(memory_format=torch.channels_last)
        teacher_depth = teacher_depth.to(memory_format=torch.channels_last)
        student_rgb = student_rgb.to(memory_format=torch.channels_last)
        student_depth = student_depth.to(memory_format=torch.channels_last)
    return teacher_rgb, teacher_depth, student_rgb, student_depth, peft


def train_stage2_one_epoch(
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
    teacher_rgb.eval()
    teacher_depth.encoder.eval()
    teacher_depth.peft.train()
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

        with amp_ctx():
            feat_t_depth = teacher_depth(depth)
            _, _, feat_s_rgb, _ = student_rgb(feat_t_rgb, feat_t_depth)
            _, _, feat_s_depth, _ = student_depth(feat_t_depth, feat_t_rgb)
            loss = loss_distil(feat_s_rgb, feat_t_rgb) + loss_distil(feat_s_depth, feat_t_depth)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_values.append(float(loss.detach().cpu()))
    return float(np.mean(loss_values)) if loss_values else 0.0


def evaluate_stage2(args, ckpt_path: Path, out_dir: Path) -> dict:
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
        depth_peft_ckpt=None,
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


def load_existing_p1(path: Path, device: torch.device):
    payload = safe_load_ckpt(path, device)
    state_dict = payload.get("state_dict", payload)
    peft = DepthAffinePEFT().to(device)
    peft.load_state_dict(state_dict, strict=True)
    meta = {
        "kind": payload.get("kind", "DepthAffinePEFT"),
        "gain": float(payload.get("gain", peft.gain.item())),
        "bias": float(payload.get("bias", peft.bias.item())),
        "created_at": payload.get("created_at"),
        "train_good": payload.get("train_good", []),
        "test_good": payload.get("test_good", []),
    }
    return peft, meta


def read_cfca_repair_final(summary_path: Path) -> float:
    with open(summary_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("scheme") == "cf_ca":
                return float(row["fusion_auroc"])
    raise RuntimeError(f"Could not find cf_ca row in {summary_path}")


def run_p1(args, device, amp_dtype, layout, stats_dataset, test_dataset, good_ids, broken_ids):
    amp_ctx = amp_context_factory(device, amp_dtype)
    teacher_rgb, teacher_depth, student_rgb, student_depth, ckpt = build_p1_models(args, device)
    stats_loader = DataLoader(
        stats_dataset,
        batch_size=args.p1_stats_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs(args.num_workers, device),
    )
    log("[P1] computing train-domain depth feature stats")
    mu_ref, var_ref = compute_depth_feature_stats(
        teacher_depth, stats_loader, device=device, amp_context_factory=amp_ctx
    )
    torch.save({"mu": [t.cpu() for t in mu_ref], "var": [t.cpu() for t in var_ref]}, layout["root"] / "reference_stats.pt")

    all_test_indices = indices_for_frames(test_dataset, set(good_ids + broken_ids))
    all_test_loader = make_loader(test_dataset, all_test_indices, args.eval_batch_size, args.num_workers, device, shuffle=False)
    baseline_auroc, baseline_frames, baseline_scores, baseline_labels = evaluate_p1_detailed(
        teacher_rgb, teacher_depth, student_rgb, student_depth, all_test_loader, device, amp_ctx, args.channels_last
    )
    write_scores_csv(layout["root"] / "baseline_scores.csv", baseline_frames, baseline_scores, baseline_labels)

    fold_parts = [list(part) for part in np.array_split(np.array(good_ids), args.folds)]
    summary_rows = [
        {
            "run_type": "baseline",
            "fold": "",
            "auroc": f"{baseline_auroc:.8f}",
            "gain": "",
            "bias": "",
            "peft_ckpt": "",
            "scores_csv": str(layout["root"] / "baseline_scores.csv"),
            "train_good": "",
            "test_good": "",
        }
    ]
    fold_records = []

    for fold_idx, test_good in enumerate(fold_parts, start=1):
        train_good = [frame_id for frame_id in good_ids if frame_id not in set(test_good)]
        train_indices = indices_for_frames(test_dataset, set(train_good))
        eval_indices = indices_for_frames(test_dataset, set(test_good + broken_ids))
        train_loader = make_loader(test_dataset, train_indices, args.p1_batch_size, args.num_workers, device, shuffle=True)
        eval_loader = make_loader(test_dataset, eval_indices, args.eval_batch_size, args.num_workers, device, shuffle=False)
        log(f"[P1] fold {fold_idx}/{args.folds}: train_good={len(train_good)}, heldout_good={len(test_good)}")
        peft, history = train_p1(
            teacher_depth,
            train_loader,
            mu_ref,
            var_ref,
            epochs=args.p1_epochs,
            lr=args.p1_lr,
            log_every=args.log_every,
            device=device,
            channels_last=args.channels_last,
            amp_ctx_factory=amp_ctx,
        )
        fold_dir = layout["cv_root"] / f"fold{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        peft_ckpt = fold_dir / f"peft_cam{args.view_id}.pth"
        save_peft(peft_ckpt, peft, args, train_good=train_good, test_good=test_good)
        write_history_csv(fold_dir / "history.csv", history, ["epoch", "loss", "gain", "bias"])

        depth_branch = DepthEncoderWithPEFT(teacher_depth, peft).to(device).eval()
        auroc, frames, scores, labels = evaluate_p1_detailed(
            teacher_rgb, depth_branch, student_rgb, student_depth, eval_loader, device, amp_ctx, args.channels_last
        )
        scores_csv = fold_dir / "scores.csv"
        write_scores_csv(scores_csv, frames, scores, labels)
        record = {
            "run_type": "p1_cv",
            "fold": fold_idx,
            "auroc": f"{float(auroc):.8f}",
            "gain": f"{peft.gain.item():.8f}",
            "bias": f"{peft.bias.item():.8f}",
            "peft_ckpt": str(peft_ckpt),
            "scores_csv": str(scores_csv),
            "train_good": " ".join(train_good),
            "test_good": " ".join(test_good),
        }
        fold_records.append(record)
        summary_rows.append(record)

    final_train_indices = indices_for_frames(test_dataset, set(good_ids))
    final_train_loader = make_loader(test_dataset, final_train_indices, args.p1_batch_size, args.num_workers, device, shuffle=True)
    log("[P1] final train on all good images")
    final_peft, final_history = train_p1(
        teacher_depth,
        final_train_loader,
        mu_ref,
        var_ref,
        epochs=args.p1_epochs,
        lr=args.p1_lr,
        log_every=args.log_every,
        device=device,
        channels_last=args.channels_last,
        amp_ctx_factory=amp_ctx,
    )
    save_peft(layout["final_ckpt"], final_peft, args, train_good=good_ids)
    write_history_csv(layout["final_history"], final_history, ["epoch", "loss", "gain", "bias"])
    final_branch = DepthEncoderWithPEFT(teacher_depth, final_peft).to(device).eval()
    final_auroc, final_frames, final_scores, final_labels = evaluate_p1_detailed(
        teacher_rgb, final_branch, student_rgb, student_depth, all_test_loader, device, amp_ctx, args.channels_last
    )
    final_scores_csv = layout["final_root"] / "scores.csv"
    write_scores_csv(final_scores_csv, final_frames, final_scores, final_labels)
    final_record = {
        "run_type": "p1_final_train_all_good",
        "fold": "final",
        "auroc": f"{float(final_auroc):.8f}",
        "gain": f"{final_peft.gain.item():.8f}",
        "bias": f"{final_peft.bias.item():.8f}",
        "peft_ckpt": str(layout["final_ckpt"]),
        "scores_csv": str(final_scores_csv),
        "train_good": " ".join(good_ids),
        "test_good": "",
    }
    summary_rows.append(final_record)
    write_csv(
        layout["summary_csv"],
        summary_rows,
        ["run_type", "fold", "auroc", "gain", "bias", "peft_ckpt", "scores_csv", "train_good", "test_good"],
    )
    with open(layout["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    with open(layout["metadata_json"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "retrained_p1",
                "baseline_auroc": baseline_auroc,
                "baseline_ckpt": args.ckpt,
                "final_gain": float(final_peft.gain.item()),
                "final_bias": float(final_peft.bias.item()),
                "ckpt_epoch": ckpt.get("epoch"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return {
        "baseline_auroc": baseline_auroc,
        "final_peft": final_peft,
        "final_gain": float(final_peft.gain.item()),
        "final_bias": float(final_peft.bias.item()),
        "summary_rows": summary_rows,
        "fold_records": fold_records,
        "final_record": final_record,
    }


def run_stage2(args, device, amp_dtype, layout, test_dataset, good_ids, broken_ids, peft_state_dict, peft_source_meta):
    baseline_ckpt_cpu = safe_load_ckpt(args.ckpt, torch.device("cpu"))
    fold_parts = [list(part) for part in np.array_split(np.array(good_ids), args.folds)]
    summary_rows = []
    p1_init = DepthAffinePEFT()
    p1_init.load_state_dict(peft_state_dict, strict=True)
    init_gain = float(p1_init.gain.item())
    init_bias = float(p1_init.bias.item())

    for fold_idx, test_good in enumerate(fold_parts, start=1):
        log(f"[Stage2] fold {fold_idx}/{args.folds}: building models")
        teacher_rgb, teacher_depth, student_rgb, student_depth, peft = build_stage2_models(
            args, device, baseline_ckpt_cpu, peft_state_dict
        )
        freeze_all(student_rgb)
        freeze_all(student_depth)
        trainable_rgb = unfreeze_cfca(student_rgb)
        trainable_depth = unfreeze_cfca(student_depth)
        peft_names = ["peft.gain", "peft.bias"]
        optimizer_params = (
            list(teacher_depth.peft.parameters())
            + [p for p in student_rgb.parameters() if p.requires_grad]
            + [p for p in student_depth.parameters() if p.requires_grad]
        )
        optimizer = torch.optim.Adam(optimizer_params, lr=args.stage2_lr)

        train_good = [fid for fid in good_ids if fid not in set(test_good)]
        train_indices = indices_for_frames(test_dataset, set(train_good))
        train_loader = make_loader(test_dataset, train_indices, args.stage2_batch_size, args.num_workers, device, shuffle=True)

        fold_dir = layout["cv_root"] / f"fold{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        history = []
        for epoch in range(1, args.stage2_epochs + 1):
            avg_loss = train_stage2_one_epoch(
                teacher_rgb, teacher_depth, student_rgb, student_depth,
                train_loader, optimizer, device, amp_dtype, args.channels_last,
            )
            if epoch == 1 or epoch == args.stage2_epochs or epoch % args.log_every == 0:
                history.append(
                    {
                        "epoch": epoch,
                        "loss": f"{avg_loss:.8f}",
                        "stage1_gain": f"{init_gain:.8f}",
                        "stage1_bias": f"{init_bias:.8f}",
                        "stage2_init_gain": f"{init_gain:.8f}",
                        "stage2_init_bias": f"{init_bias:.8f}",
                        "stage2_current_gain": f"{teacher_depth.peft.gain.item():.8f}",
                        "stage2_current_bias": f"{teacher_depth.peft.bias.item():.8f}",
                    }
                )
                log(
                    f"[Stage2] fold {fold_idx}: epoch {epoch}/{args.stage2_epochs}, "
                    f"loss={avg_loss:.8f}, gain={teacher_depth.peft.gain.item():.8f}, bias={teacher_depth.peft.bias.item():.8f}"
                )

        fold_ckpt = fold_dir / f"peft_full_then_cf_ca_cam{args.view_id}.pth"
        save_joint_ckpt(
            fold_ckpt,
            student_rgb,
            student_depth,
            teacher_depth.peft,
            epoch=args.stage2_epochs,
            best_val_loss=float(history[-1]["loss"]) if history else None,
        )
        write_history_csv(
            fold_dir / "history.csv",
            history,
            [
                "epoch",
                "loss",
                "stage1_gain",
                "stage1_bias",
                "stage2_init_gain",
                "stage2_init_bias",
                "stage2_current_gain",
                "stage2_current_bias",
            ],
        )
        eval_result = evaluate_stage2(args, fold_ckpt, fold_dir / "eval")
        summary_rows.append(
            {
                "run_type": "peft_full_then_cf_ca_cv",
                "fold": fold_idx,
                "rgb_auroc": f"{float(eval_result['auroc_by_source']['rgb']):.8f}",
                "depth_auroc": f"{float(eval_result['auroc_by_source']['depth']):.8f}",
                "fusion_auroc": f"{float(eval_result['auroc_by_source']['fusion']):.8f}",
                "delta_vs_peft_full": f"{float(eval_result['auroc_by_source']['fusion']) - PEFT_FULL_FUSION:.8f}",
                "stage1_gain": f"{init_gain:.8f}",
                "stage1_bias": f"{init_bias:.8f}",
                "stage2_init_gain": f"{init_gain:.8f}",
                "stage2_init_bias": f"{init_bias:.8f}",
                "stage2_final_gain": f"{teacher_depth.peft.gain.item():.8f}",
                "stage2_final_bias": f"{teacher_depth.peft.bias.item():.8f}",
                "optimizer_param_names": " ".join(peft_names + trainable_rgb + trainable_depth),
                "train_good": " ".join(train_good),
                "test_good": " ".join(test_good),
                "ckpt": str(fold_ckpt),
            }
        )

    teacher_rgb, teacher_depth, student_rgb, student_depth, peft = build_stage2_models(
        args, device, baseline_ckpt_cpu, peft_state_dict
    )
    freeze_all(student_rgb)
    freeze_all(student_depth)
    trainable_rgb = unfreeze_cfca(student_rgb)
    trainable_depth = unfreeze_cfca(student_depth)
    peft_names = ["peft.gain", "peft.bias"]
    optimizer_params = (
        list(teacher_depth.peft.parameters())
        + [p for p in student_rgb.parameters() if p.requires_grad]
        + [p for p in student_depth.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.Adam(optimizer_params, lr=args.stage2_lr)
    final_train_indices = indices_for_frames(test_dataset, set(good_ids))
    final_loader = make_loader(test_dataset, final_train_indices, args.stage2_batch_size, args.num_workers, device, shuffle=True)
    final_history = []
    log("[Stage2] final train on all good images")
    for epoch in range(1, args.stage2_epochs + 1):
        avg_loss = train_stage2_one_epoch(
            teacher_rgb, teacher_depth, student_rgb, student_depth,
            final_loader, optimizer, device, amp_dtype, args.channels_last,
        )
        if epoch == 1 or epoch == args.stage2_epochs or epoch % args.log_every == 0:
            final_history.append(
                {
                    "epoch": epoch,
                    "loss": f"{avg_loss:.8f}",
                    "stage1_gain": f"{init_gain:.8f}",
                    "stage1_bias": f"{init_bias:.8f}",
                    "stage2_init_gain": f"{init_gain:.8f}",
                    "stage2_init_bias": f"{init_bias:.8f}",
                    "stage2_current_gain": f"{teacher_depth.peft.gain.item():.8f}",
                    "stage2_current_bias": f"{teacher_depth.peft.bias.item():.8f}",
                }
            )
            log(
                f"[Stage2] final: epoch {epoch}/{args.stage2_epochs}, "
                f"loss={avg_loss:.8f}, gain={teacher_depth.peft.gain.item():.8f}, bias={teacher_depth.peft.bias.item():.8f}"
            )

    save_joint_ckpt(
        layout["final_ckpt"],
        student_rgb,
        student_depth,
        teacher_depth.peft,
        epoch=args.stage2_epochs,
        best_val_loss=float(final_history[-1]["loss"]) if final_history else None,
    )
    write_history_csv(
        layout["final_history"],
        final_history,
        [
            "epoch",
            "loss",
            "stage1_gain",
            "stage1_bias",
            "stage2_init_gain",
            "stage2_init_bias",
            "stage2_current_gain",
            "stage2_current_bias",
        ],
    )
    final_eval = evaluate_stage2(args, layout["final_ckpt"], layout["final_root"] / "eval")
    final_row = {
        "run_type": "peft_full_then_cf_ca_final",
        "fold": "final",
        "rgb_auroc": f"{float(final_eval['auroc_by_source']['rgb']):.8f}",
        "depth_auroc": f"{float(final_eval['auroc_by_source']['depth']):.8f}",
        "fusion_auroc": f"{float(final_eval['auroc_by_source']['fusion']):.8f}",
        "delta_vs_peft_full": f"{float(final_eval['auroc_by_source']['fusion']) - PEFT_FULL_FUSION:.8f}",
        "stage1_gain": f"{init_gain:.8f}",
        "stage1_bias": f"{init_bias:.8f}",
        "stage2_init_gain": f"{init_gain:.8f}",
        "stage2_init_bias": f"{init_bias:.8f}",
        "stage2_final_gain": f"{teacher_depth.peft.gain.item():.8f}",
        "stage2_final_bias": f"{teacher_depth.peft.bias.item():.8f}",
        "optimizer_param_names": " ".join(peft_names + trainable_rgb + trainable_depth),
        "train_good": " ".join(good_ids),
        "test_good": "",
        "ckpt": str(layout["final_ckpt"]),
    }
    summary_rows.append(final_row)
    write_csv(
        layout["summary_csv"],
        summary_rows,
        [
            "run_type",
            "fold",
            "rgb_auroc",
            "depth_auroc",
            "fusion_auroc",
            "delta_vs_peft_full",
            "stage1_gain",
            "stage1_bias",
            "stage2_init_gain",
            "stage2_init_bias",
            "stage2_final_gain",
            "stage2_final_bias",
            "optimizer_param_names",
            "train_good",
            "test_good",
            "ckpt",
        ],
    )
    with open(layout["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    with open(layout["metadata_json"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "peft_full_then_cf_ca",
                "baseline_ckpt": args.ckpt,
                "p1_source": peft_source_meta,
                "optimizer_param_names": peft_names + trainable_rgb + trainable_depth,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return {"summary_rows": summary_rows, "final_row": final_row}


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.precision, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / f"cam{args.view_id}_peft_full_then_cf_ca_{timestamp}"
    layout = build_run_layout(output_root)
    p1_layout = stage_layout(layout["p1_root"], f"final_peft_cam{args.view_id}.pth")
    stage2_layout = stage_layout(layout["stage2_root"], f"peft_full_then_cf_ca_cam{args.view_id}.pth")

    log(f"ROOT={_PROJ_ROOT}")
    log(f"DEVICE={device}")
    log(f"OUTPUT={output_root}")

    stats_dataset = build_stats_dataset(args)
    test_dataset = build_test_dataset(args)
    good_ids = sorted([s["frame_id"] for s in test_dataset.samples if int(s["label"]) == 0])
    broken_ids = sorted([s["frame_id"] for s in test_dataset.samples if int(s["label"]) == 1])
    if len(good_ids) < args.folds:
        raise RuntimeError(f"Need at least {args.folds} good images, got {len(good_ids)}")
    log(
        f"Dataset ready: good={len(good_ids)}, broken={len(broken_ids)}, "
        f"patches_per_image={test_dataset.num_patches}"
    )

    if args.reuse_existing_p1:
        log(f"[P1] reusing existing ckpt: {args.existing_p1_ckpt}")
        peft, p1_meta = load_existing_p1(Path(args.existing_p1_ckpt), device)
        p1_state_dict = {k: v.detach().cpu() for k, v in peft.state_dict().items()}
        p1_summary_rows = [
            {
                "run_type": "p1_reused_final",
                "fold": "final",
                "auroc": f"{PEFT_FULL_FUSION:.8f}",
                "gain": f"{peft.gain.item():.8f}",
                "bias": f"{peft.bias.item():.8f}",
                "peft_ckpt": str(args.existing_p1_ckpt),
                "scores_csv": "",
                "train_good": " ".join(p1_meta.get("train_good", [])),
                "test_good": " ".join(p1_meta.get("test_good", [])),
            }
        ]
        write_csv(
            p1_layout["summary_csv"],
            p1_summary_rows,
            ["run_type", "fold", "auroc", "gain", "bias", "peft_ckpt", "scores_csv", "train_good", "test_good"],
        )
        with open(p1_layout["summary_json"], "w", encoding="utf-8") as f:
            json.dump(p1_summary_rows, f, ensure_ascii=False, indent=2)
        with open(p1_layout["metadata_json"], "w", encoding="utf-8") as f:
            json.dump({"source": "reused_existing_p1", "existing_p1_ckpt": args.existing_p1_ckpt, "meta": p1_meta}, f, ensure_ascii=False, indent=2)
    else:
        p1_result = run_p1(args, device, amp_dtype, p1_layout, stats_dataset, test_dataset, good_ids, broken_ids)
        p1_state_dict = {k: v.detach().cpu() for k, v in p1_result["final_peft"].state_dict().items()}
        p1_meta = {
            "source": "retrained_p1",
            "final_gain": p1_result["final_gain"],
            "final_bias": p1_result["final_bias"],
            "summary_csv": str(p1_layout["summary_csv"]),
        }

    stage2_result = run_stage2(
        args,
        device,
        amp_dtype,
        stage2_layout,
        test_dataset,
        good_ids,
        broken_ids,
        p1_state_dict,
        p1_meta,
    )

    cfca_repair_final = read_cfca_repair_final(Path(args.cfca_repair_summary))
    comparison_rows = [
        {
            "scheme": "baseline_full",
            "rgb_auroc": "",
            "depth_auroc": "",
            "fusion_auroc": "0.35000000",
            "delta_vs_peft_full": f"{0.3500 - PEFT_FULL_FUSION:.8f}",
            "delta_vs_cf_ca_repair": f"{0.3500 - cfca_repair_final:.8f}",
            "ckpt": args.ckpt,
        },
        {
            "scheme": "peft_full",
            "rgb_auroc": "0.60000000",
            "depth_auroc": "0.67500000",
            "fusion_auroc": f"{PEFT_FULL_FUSION:.8f}",
            "delta_vs_peft_full": "0.00000000",
            "delta_vs_cf_ca_repair": f"{PEFT_FULL_FUSION - cfca_repair_final:.8f}",
            "ckpt": args.existing_p1_ckpt if args.reuse_existing_p1 else str(p1_layout["final_ckpt"]),
        },
        {
            "scheme": "cf_ca_repair_final",
            "rgb_auroc": "0.67500000",
            "depth_auroc": "0.80000000",
            "fusion_auroc": f"{cfca_repair_final:.8f}",
            "delta_vs_peft_full": f"{cfca_repair_final - PEFT_FULL_FUSION:.8f}",
            "delta_vs_cf_ca_repair": "0.00000000",
            "ckpt": "",
        },
        {
            "scheme": "peft_full_then_cf_ca_final",
            "rgb_auroc": stage2_result["final_row"]["rgb_auroc"],
            "depth_auroc": stage2_result["final_row"]["depth_auroc"],
            "fusion_auroc": stage2_result["final_row"]["fusion_auroc"],
            "delta_vs_peft_full": stage2_result["final_row"]["delta_vs_peft_full"],
            "delta_vs_cf_ca_repair": f"{float(stage2_result['final_row']['fusion_auroc']) - cfca_repair_final:.8f}",
            "ckpt": stage2_result["final_row"]["ckpt"],
        },
    ]
    write_csv(
        layout["comparison_csv"],
        comparison_rows,
        ["scheme", "rgb_auroc", "depth_auroc", "fusion_auroc", "delta_vs_peft_full", "delta_vs_cf_ca_repair", "ckpt"],
    )
    write_csv(
        layout["summary_csv"],
        comparison_rows,
        ["scheme", "rgb_auroc", "depth_auroc", "fusion_auroc", "delta_vs_peft_full", "delta_vs_cf_ca_repair", "ckpt"],
    )
    with open(layout["summary_json"], "w", encoding="utf-8") as f:
        json.dump(comparison_rows, f, ensure_ascii=False, indent=2)
    with open(layout["metadata_json"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "baseline_ckpt": args.ckpt,
                "reuse_existing_p1": args.reuse_existing_p1,
                "existing_p1_ckpt": args.existing_p1_ckpt,
                "cfca_repair_summary": args.cfca_repair_summary,
                "p1_meta": p1_meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    log(f"Comparison summary: {layout['comparison_csv']}")
    log(f"Stage2 final fusion AUROC: {stage2_result['final_row']['fusion_auroc']}")


if __name__ == "__main__":
    main()
