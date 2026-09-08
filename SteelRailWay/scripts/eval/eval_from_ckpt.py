#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立评估脚本：从已有 best ckpt 加载并在测试集上跑 AUROC。

用途：
    - 训练崩在最后评估阶段时，可用本脚本接力评估，避免重新训练
    - 修改测试集后只跑评估
    - 比较不同 epoch 的 ckpt

用法：
    python scripts/eval/eval_from_ckpt.py \
        --ckpt outputs/rail/20260501_xxxxx_cam1_xxx/best_cam1.pth \
        --train_root /data1/Leaddo_data/20260327-resize512 \
        --test_root  ./rail_mvtec_gt_test \
        --view_id 1
"""

# >>> path-bootstrap >>>
import os, sys
from pathlib import Path
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
# <<< path-bootstrap <<<

import argparse
import csv
import json
from contextlib import nullcontext
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import numpy as np

from datasets.rail_dataset import RailDualModalDataset
from models.trd.encoder import ResNet50Encoder
from models.trd.decoder import ResNet50DualModalDecoder
from eval.eval_utils import cal_anomaly_map
from rail_cad.registry import load_registry, resolve_active_peft_ckpt, resolve_artifact_path
from rail_peft import (
    DepthAffinePEFT,
    DepthEncoderWithPEFT,
    compute_teacher_feature_means,
    expand_feature_means,
    zeros_like_feature_list,
)
from sklearn.metrics import roc_auc_score

MODULE_ABLATION_MODES = ("full", "no_cf", "no_ca", "no_cf_ca")
CROSS_SOURCES = ("rgb", "depth", "fusion")
ISOLATED_SOURCES = ("rgb_isolated", "depth_isolated")
SCORE_SOURCES = CROSS_SOURCES + ISOLATED_SOURCES
FUSION_RULES = ("sum", "max_norm")
SOURCE_SEMANTICS = {
    "rgb": "RGB branch score (cross-conditioned by current depth teacher features)",
    "depth": "Depth branch score (cross-conditioned by current RGB teacher features)",
    "fusion": "Fusion score from current rgb_cross + depth_cross anomaly maps",
    "rgb_isolated": "RGB branch score with depth assist replaced by a fixed training-domain reference",
    "depth_isolated": "Depth branch score with RGB assist replaced by a fixed training-domain reference",
}


try:
    from torch.amp import autocast as _autocast_new

    def autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        return _autocast_new(device_type, enabled=enabled, dtype=dtype)

except ImportError:
    try:
        from torch.cuda.amp import autocast as _autocast_old

        def autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            return _autocast_old(enabled=enabled, dtype=dtype)

    except ImportError:
        autocast = None


def safe_load_ckpt(path, device):
    """兼容 PyTorch 2.6+ 的 weights_only 默认 True：显式关掉。
    本工程的 ckpt 由自己脚本生成，来源可信。
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def strip_prefix(sd):
    """去掉 torch.compile 包装后的 _orig_mod. 前缀"""
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def load_depth_peft(path, device):
    """Load a DepthAffinePEFT checkpoint saved either as a raw state_dict or payload."""
    payload = safe_load_ckpt(path, device)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    else:
        state_dict = payload
        metadata = {}

    if not isinstance(state_dict, dict):
        raise TypeError(f"Invalid depth PEFT checkpoint: {path}")

    state_dict = {
        k.replace("_orig_mod.", "").replace("module.", "").replace("peft.", ""): v
        for k, v in state_dict.items()
        if k.replace("_orig_mod.", "").replace("module.", "").replace("peft.", "") in {"gain", "bias"}
    }
    peft = DepthAffinePEFT().to(device)
    peft.load_state_dict(state_dict, strict=True)
    peft.eval()
    for param in peft.parameters():
        param.requires_grad_(False)
    return peft, metadata


def load_depth_peft_from_joint_ckpt(payload_or_path, device):
    """Load DepthAffinePEFT params from a joint model checkpoint if present."""
    payload = (
        safe_load_ckpt(payload_or_path, device)
        if isinstance(payload_or_path, (str, Path))
        else payload_or_path
    )
    if not isinstance(payload, dict):
        return None, {}

    if "depth_peft" in payload and isinstance(payload["depth_peft"], dict):
        state_dict = payload["depth_peft"]
        metadata = {
            "source": "joint_ckpt.depth_peft",
            "gain": float(payload.get("depth_peft_gain")) if payload.get("depth_peft_gain") is not None else None,
            "bias": float(payload.get("depth_peft_bias")) if payload.get("depth_peft_bias") is not None else None,
        }
    else:
        state_dict = {
            k.replace("_orig_mod.", "").replace("module.", "").replace("peft.", "").replace("depth_peft.", ""): v
            for k, v in payload.items()
            if k.replace("_orig_mod.", "").replace("module.", "").replace("peft.", "").replace("depth_peft.", "") in {"gain", "bias"}
        }
        metadata = {"source": "joint_ckpt.flat_keys"} if state_dict else {}

    if not state_dict:
        return None, {}

    peft = DepthAffinePEFT().to(device)
    peft.load_state_dict(state_dict, strict=True)
    peft.eval()
    for param in peft.parameters():
        param.requires_grad_(False)
    return peft, metadata


def resolve_depth_peft_from_cad_registry(cad_registry, *, task_id=None, view_id: int | None = None):
    if not cad_registry:
        return None, {}
    registry = load_registry(cad_registry)
    resolved = resolve_active_peft_ckpt(registry, task_id=task_id, view_id=view_id)
    if not resolved:
        return None, {
            "source": "cad_registry",
            "cad_registry": str(resolve_artifact_path(cad_registry)),
            "matched": False,
        }
    peft_path = resolve_artifact_path(resolved)
    return peft_path, {
        "source": "cad_registry",
        "cad_registry": str(resolve_artifact_path(cad_registry)),
        "matched": True,
        "task_id": str(task_id if task_id is not None else view_id),
    }


def resolve_amp_dtype(precision, device):
    """评估阶段的 AMP 配置，与训练脚本保持一致。"""
    if device.type != "cuda" or autocast is None:
        return None
    if precision == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if precision in {"bf16", "fp16"}:
        return torch.float16
    return None


def format_metric(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def format_metric_map(metric_map, sources=SCORE_SOURCES):
    parts = []
    for source in sources:
        parts.append(f"{source}={format_metric(metric_map.get(source))}")
    return ", ".join(parts)


def write_scores_csv(path, frame_ids, scores, labels):
    order = np.argsort(-scores)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "frame_id", "score", "label"])
        writer.writeheader()
        for rank, idx in enumerate(order):
            writer.writerow({
                "rank": rank,
                "frame_id": frame_ids[idx],
                "score": f"{float(scores[idx]):.8f}",
                "label": int(labels[idx]),
            })


def default_scores_csv_path(ckpt_path, score_source):
    ckpt_path = str(ckpt_path)
    if score_source == "fusion":
        return ckpt_path.replace(".pth", "_test_scores.csv")
    return ckpt_path.replace(".pth", f"_test_scores_{score_source}.csv")


def batch_map_to_scores(amap):
    if amap.ndim == 3:
        return amap.reshape(amap.shape[0], -1).max(axis=1)
    return np.array([amap.max()])


def amp_context_factory(device, amp_dtype):
    if amp_dtype is None or autocast is None:
        return nullcontext
    return lambda: autocast(device.type, enabled=True, dtype=amp_dtype)


def assist_stats_filename(view_id, modality):
    return f"cam{int(view_id)}_{modality}_train_mean.pt"


def assist_stats_path(stats_dir, view_id, modality):
    return Path(stats_dir) / assist_stats_filename(view_id, modality)


def save_feature_mean_payload(path, feature_means, view_id, modality):
    payload = {
        "view_id": int(view_id),
        "modality": str(modality),
        "feature_means": [tensor.detach().cpu() for tensor in feature_means],
    }
    torch.save(payload, path)


def load_feature_mean_payload(path, device):
    payload = safe_load_ckpt(path, device)
    if isinstance(payload, dict) and "feature_means" in payload:
        feature_means = payload["feature_means"]
    elif isinstance(payload, list):
        feature_means = payload
    else:
        raise TypeError(f"Invalid feature-mean payload: {path}")
    return [tensor.to(device=device, dtype=torch.float32) for tensor in feature_means]


def build_assist_feature_list(modality, like_feats, assist_fill, assist_feature_means):
    if assist_fill == "zeros":
        return zeros_like_feature_list(like_feats)
    if assist_fill != "train_mean":
        raise ValueError(f"Unsupported assist_fill: {assist_fill}")
    if not assist_feature_means or modality not in assist_feature_means:
        raise RuntimeError(f"Missing train-mean assist features for modality: {modality}")
    return expand_feature_means(assist_feature_means[modality], like_feats)


def resolve_train_sample_num(args):
    value = getattr(args, "train_sample_num", None)
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def build_assist_stats_dataset(args):
    return RailDualModalDataset(
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        split="train",
        img_size=args.img_size,
        depth_norm=args.depth_norm,
        use_patch=False,
        train_sample_ratio=float(getattr(args, "train_sample_ratio", 1.0)),
        train_sample_num=resolve_train_sample_num(args),
        random_seed=int(getattr(args, "train_sample_seed", 42)),
        train_val_test_split=[1.0, 0.0, 0.0],
        sampling_mode=str(getattr(args, "sampling_mode", "uniform_time")),
        preload=args.preload,
        preload_workers=args.preload_workers,
    )


def build_assist_stats_loader(dataset, batch_size, num_workers, device):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
        "persistent_workers": (num_workers > 0),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        **kwargs,
    )


def ensure_assist_feature_means(
    args,
    device,
    amp_dtype,
    teacher_rgb,
    teacher_depth,
):
    assist_fill = getattr(args, "assist_fill", "train_mean")
    assist_stats_dir = getattr(args, "assist_stats_dir", None)
    if assist_fill == "zeros":
        return {"rgb": None, "depth": None}

    if assist_fill != "train_mean":
        raise ValueError(f"Unsupported assist_fill: {assist_fill}")

    stats_dir = Path(assist_stats_dir) if assist_stats_dir else None
    if stats_dir:
        stats_dir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    missing_modalities = []
    for modality in ("rgb", "depth"):
        path = assist_stats_path(stats_dir, args.view_id, modality) if stats_dir else None
        if path and path.exists():
            print(f"Loading {modality} train-mean assist features: {path}")
            loaded[modality] = load_feature_mean_payload(path, device)
        else:
            missing_modalities.append(modality)

    if not missing_modalities:
        return loaded

    print(
        "Assist stats sampling: "
        f"train_sample_num={resolve_train_sample_num(args)}, "
        f"train_sample_ratio={getattr(args, 'train_sample_ratio', 1.0)}, "
        f"sampling_mode={getattr(args, 'sampling_mode', 'uniform_time')}, "
        f"train_sample_seed={getattr(args, 'train_sample_seed', 42)}"
    )
    print(f"Computing train-mean assist features for Cam{args.view_id}: {', '.join(missing_modalities)}")
    stats_dataset = build_assist_stats_dataset(args)
    if len(stats_dataset) == 0:
        raise RuntimeError(f"Cannot compute assist stats: empty train split for Cam{args.view_id}")
    assist_stats_batch_size = int(
        getattr(args, "assist_stats_batch_size", max(8, min(32, int(getattr(args, "batch_size", 8)))))
    )
    stats_loader = build_assist_stats_loader(
        stats_dataset,
        batch_size=assist_stats_batch_size,
        num_workers=int(args.num_workers),
        device=device,
    )
    amp_ctx = amp_context_factory(device, amp_dtype)

    if "rgb" in missing_modalities:
        loaded["rgb"] = compute_teacher_feature_means(
            teacher_rgb,
            stats_loader,
            input_key="intensity",
            device=device,
            amp_context_factory=amp_ctx,
            channels_last=bool(getattr(args, "channels_last", False)),
        )
        if stats_dir:
            rgb_path = assist_stats_path(stats_dir, args.view_id, "rgb")
            save_feature_mean_payload(rgb_path, loaded["rgb"], args.view_id, "rgb")
            print(f"Saved rgb train-mean assist features: {rgb_path}")

    if "depth" in missing_modalities:
        loaded["depth"] = compute_teacher_feature_means(
            teacher_depth,
            stats_loader,
            input_key="depth",
            device=device,
            amp_context_factory=amp_ctx,
            channels_last=bool(getattr(args, "channels_last", False)),
        )
        if stats_dir:
            depth_path = assist_stats_path(stats_dir, args.view_id, "depth")
            save_feature_mean_payload(depth_path, loaded["depth"], args.view_id, "depth")
            print(f"Saved depth train-mean assist features: {depth_path}")

    return loaded


def append_eval_log(log_file, result):
    """把补评估结果追加到训练 run 的 training.log。"""
    if not log_file:
        return

    status = result["status"]
    reason = result.get("reason", "")
    auroc = format_metric(result.get("auroc"))
    best_val_loss = result.get("best_val_loss")
    best_val_loss_text = (
        f"{best_val_loss:.4f}" if isinstance(best_val_loss, (int, float)) else "N/A"
    )

    msg = "\n"
    msg += "=" * 50 + "\n"
    msg += "Post-hoc evaluation on test set (best checkpoint)\n"
    msg += "=" * 50 + "\n"
    msg += f"Evaluation time: {result['evaluated_at']}\n"
    msg += f"Script: scripts/eval/eval_from_ckpt.py\n"
    msg += f"Checkpoint: {result['ckpt']}\n"
    if result.get("depth_peft_ckpt"):
        msg += f"Depth PEFT: {result['depth_peft_ckpt']}\n"
    msg += f"Module ablation: {result.get('module_ablation', 'full')}\n"
    msg += f"Score source: {result.get('score_source_selected', 'fusion')}\n"
    msg += f"Fusion rule: {result.get('fusion_rule', 'sum')}\n"
    msg += f"Status: {status}"
    if reason:
        msg += f" ({reason})"
    msg += "\n"
    msg += f"Final Result - Cam {result['view_id']}\n"
    msg += f"  Best epoch: {result.get('best_epoch', 'N/A')}\n"
    msg += f"  Best Val Loss: {best_val_loss_text}\n"
    msg += f"  Test AUROC: {auroc}\n"
    if result.get("auroc_by_source"):
        msg += f"  AUROC by source: {format_metric_map(result['auroc_by_source'])}\n"
    msg += f"  #patches: {result['num_patches']}\n"
    msg += f"  #images: {result['num_images']}\n"
    msg += f"  #abnormal: {result['num_abnormal']}, #normal: {result['num_normal']}\n"
    if result.get("score_min") is not None:
        msg += f"  score range: [{result['score_min']:.4f}, {result['score_max']:.4f}]\n"
    if result.get("scores_csv"):
        msg += f"  Scores CSV: {result['scores_csv']}\n"
    msg += "=" * 50 + "\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg)


@torch.no_grad()
def evaluate(
    teacher_rgb,
    teacher_depth,
    student_rgb,
    student_depth,
    test_loader,
    device,
    amp_dtype=None,
    channels_last=False,
    assist_fill="train_mean",
    assist_feature_means=None,
    fusion_rule="sum",
):
    """与 train_trd_rail.py 中的 evaluate 一致：patch 分数 max 聚合到原图。"""
    student_rgb.eval()
    student_depth.eval()

    img_scores = {source: {} for source in SCORE_SOURCES}
    img_labels = {}
    rgb_global_min = None
    rgb_global_max = None
    depth_global_min = None
    depth_global_max = None

    for data in test_loader:
        rgb = data["intensity"].to(device, non_blocking=True)
        depth = data["depth"].to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            rgb = rgb.contiguous(memory_format=torch.channels_last)
            depth = depth.contiguous(memory_format=torch.channels_last)
        labels = data["label"].cpu().numpy()
        frame_ids = data["frame_id"]

        amp_ctx = (
            autocast(device.type, enabled=True, dtype=amp_dtype)
            if amp_dtype is not None and autocast is not None
            else nullcontext()
        )
        with amp_ctx:
            feat_t_rgb = teacher_rgb(rgb)
            feat_t_depth = teacher_depth(depth)
            _, _, feat_s_rgb, _ = student_rgb(feat_t_rgb, feat_t_depth)
            _, _, feat_s_depth, _ = student_depth(feat_t_depth, feat_t_rgb)

            feat_t_depth_assist = build_assist_feature_list(
                "depth",
                feat_t_depth,
                assist_fill,
                assist_feature_means,
            )
            feat_t_rgb_assist = build_assist_feature_list(
                "rgb",
                feat_t_rgb,
                assist_fill,
                assist_feature_means,
            )
            _, _, feat_s_rgb_isolated, _ = student_rgb(feat_t_rgb, feat_t_depth_assist)
            _, _, feat_s_depth_isolated, _ = student_depth(feat_t_depth, feat_t_rgb_assist)

        amap_rgb, _ = cal_anomaly_map(feat_s_rgb, feat_t_rgb, out_size=(256, 256), amap_mode='mul')
        amap_depth, _ = cal_anomaly_map(feat_s_depth, feat_t_depth, out_size=(256, 256), amap_mode='mul')
        amap_rgb_isolated, _ = cal_anomaly_map(
            feat_s_rgb_isolated,
            feat_t_rgb,
            out_size=(256, 256),
            amap_mode='mul',
        )
        amap_depth_isolated, _ = cal_anomaly_map(
            feat_s_depth_isolated,
            feat_t_depth,
            out_size=(256, 256),
            amap_mode='mul',
        )
        amap_rgb = np.asarray(amap_rgb, dtype=np.float32)
        amap_depth = np.asarray(amap_depth, dtype=np.float32)
        amap_rgb_isolated = np.asarray(amap_rgb_isolated, dtype=np.float32)
        amap_depth_isolated = np.asarray(amap_depth_isolated, dtype=np.float32)
        amap_fusion_sum = amap_rgb + amap_depth

        batch_rgb_min = float(np.min(amap_rgb))
        batch_rgb_max = float(np.max(amap_rgb))
        batch_depth_min = float(np.min(amap_depth))
        batch_depth_max = float(np.max(amap_depth))
        rgb_global_min = batch_rgb_min if rgb_global_min is None else min(rgb_global_min, batch_rgb_min)
        rgb_global_max = batch_rgb_max if rgb_global_max is None else max(rgb_global_max, batch_rgb_max)
        depth_global_min = batch_depth_min if depth_global_min is None else min(depth_global_min, batch_depth_min)
        depth_global_max = batch_depth_max if depth_global_max is None else max(depth_global_max, batch_depth_max)
        amap_by_source = {
            "rgb": amap_rgb,
            "depth": amap_depth,
            "fusion": amap_fusion_sum,
            "rgb_isolated": amap_rgb_isolated,
            "depth_isolated": amap_depth_isolated,
        }
        batch_scores = {
            source: batch_map_to_scores(amap)
            for source, amap in amap_by_source.items()
        }

        for idx, (label, fid) in enumerate(zip(labels, frame_ids)):
            if fid not in img_labels:
                img_labels[fid] = int(label)
            for source in SCORE_SOURCES:
                img_scores[source].setdefault(fid, []).append(float(batch_scores[source][idx]))

    frame_ids = sorted(img_labels.keys())
    image_labels = np.array([img_labels[fid] for fid in frame_ids], dtype=np.int64)

    if len(frame_ids) == 0:
        print("Warning: no test samples, AUROC = N/A")
        return {
            "frame_ids": [],
            "labels": image_labels,
            "scores_by_source": {
                source: np.array([], dtype=np.float64) for source in SCORE_SOURCES
            },
            "auroc_by_source": {
                source: None for source in SCORE_SOURCES
            },
        }

    single_class = len(np.unique(image_labels)) < 2
    if single_class:
        print("Warning: only one class in test set, AUROC = N/A")
    scores_by_source = {}
    auroc_by_source = {}
    for source in SCORE_SOURCES:
        if source == "fusion" and fusion_rule == "max_norm":
            rgb_denom = max((rgb_global_max or 0.0) - (rgb_global_min or 0.0), 1e-8)
            depth_denom = max((depth_global_max or 0.0) - (depth_global_min or 0.0), 1e-8)
            image_scores_list = []
            for fid in frame_ids:
                fusion_patch_scores = []
                for rgb_score, depth_score in zip(img_scores["rgb"][fid], img_scores["depth"][fid]):
                    rgb_norm_score = (
                        0.0 if (rgb_global_max is None or rgb_global_max <= rgb_global_min)
                        else (rgb_score - rgb_global_min) / rgb_denom
                    )
                    depth_norm_score = (
                        0.0 if (depth_global_max is None or depth_global_max <= depth_global_min)
                        else (depth_score - depth_global_min) / depth_denom
                    )
                    fusion_patch_scores.append(max(rgb_norm_score, depth_norm_score))
                image_scores = np.array([max(fusion_patch_scores)], dtype=np.float64)
                image_scores_list.append(image_scores[0])
            image_scores = np.array(image_scores_list, dtype=np.float64)
        else:
            image_scores = np.array(
                [max(img_scores[source][fid]) for fid in frame_ids],
                dtype=np.float64,
            )
        scores_by_source[source] = image_scores
        auroc_by_source[source] = (
            None if single_class else float(roc_auc_score(image_labels, image_scores))
        )

    return {
        "frame_ids": frame_ids,
        "labels": image_labels,
        "scores_by_source": scores_by_source,
        "auroc_by_source": auroc_by_source,
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="best_camN.pth 路径")
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--test_root", type=str, required=True)
    parser.add_argument("--view_id", type=int, required=True)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--depth_norm", type=str, default="zscore")
    # 测试 patch 配置（与训练脚本默认一致）
    parser.add_argument("--use_patch", action="store_true", default=True)
    parser.add_argument("--no_patch", action="store_false", dest="use_patch")
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--preload_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--no_channels_last", action="store_false", dest="channels_last")
    parser.add_argument("--output_log", type=str, default=None,
                        help="可选：把最终评估结果追加到指定 training.log")
    parser.add_argument("--append_log", action="store_true",
                        help="配合 --output_log 使用，确认写回日志")
    parser.add_argument("--scores_csv", type=str, default=None,
                        help="可选：指定分数 CSV 输出路径，默认写到 ckpt 同目录")
    parser.add_argument("--result_json", type=str, default=None,
                        help="可选：保存结构化评估结果 JSON，便于批量汇总")
    parser.add_argument("--depth_peft_ckpt", type=str, default=None,
                        help="可选：DepthAffinePEFT checkpoint；提供时仅替换 depth teacher 调用链")
    parser.add_argument("--cad_registry", type=str, default=None,
                        help="可选：CAD registry.json 或其根目录；未显式指定 depth_peft_ckpt 时按 task_id/view_id 路由")
    parser.add_argument("--module_ablation", type=str, default="full", choices=MODULE_ABLATION_MODES,
                        help="基于已有 ckpt 的推理期模块路径消融：full/no_cf/no_ca/no_cf_ca")
    parser.add_argument("--score_source", type=str, default="fusion", choices=SCORE_SOURCES,
                        help="顶层 AUROC / scores_csv 采用的分支来源，默认 fusion")
    parser.add_argument("--fusion_rule", type=str, default="sum", choices=FUSION_RULES,
                        help="fusion 分支的图级融合规则：sum=直接相加；max_norm=各分支先按全测试集 map-level 全局 min/max 归一化后逐像素 max")
    parser.add_argument("--scores_dir", type=str, default=None,
                        help="可选：额外保存五份逐图分数 CSV 的目录")
    parser.add_argument("--assist_fill", type=str, default="train_mean",
                        choices=["train_mean", "zeros"],
                        help="isolated 分支的辅助特征替代策略，默认 train_mean")
    parser.add_argument("--assist_stats_dir", type=str, default=None,
                        help="可选：训练域辅助特征均值的缓存目录")
    parser.add_argument("--train_sample_ratio", type=float, default=1.0,
                        help="仅用于 isolated train-mean 统计的训练集采样比例")
    parser.add_argument("--train_sample_num", type=int, default=None,
                        help="仅用于 isolated train-mean 统计的训练集采样数量，优先级高于 ratio；<=0 表示使用全量")
    parser.add_argument("--sampling_mode", type=str, default="uniform_time",
                        choices=["uniform_time", "random"],
                        help="仅用于 isolated train-mean 统计的训练集采样模式")
    parser.add_argument("--train_sample_seed", type=int, default=42,
                        help="仅用于 isolated train-mean 统计的训练集采样随机种子")
    return parser


def evaluate_from_args(args):
    evaluated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.precision, device)
    print(f"Device: {device}")
    if amp_dtype is None:
        print("Precision: fp32")
    else:
        print(f"Precision: {args.precision}")
    print(f"Loading ckpt: {args.ckpt}")

    # ---- 1. 构建测试集 ----
    test_dataset = RailDualModalDataset(
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        split="test",
        img_size=args.img_size,
        depth_norm=args.depth_norm,
        use_patch=args.use_patch,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        preload=args.preload,
        preload_workers=args.preload_workers,
    )
    print(f"Test samples: {len(test_dataset)}")

    result = {
        "evaluated_at": evaluated_at,
        "view_id": int(args.view_id),
        "ckpt": str(args.ckpt),
        "depth_peft_ckpt": str(getattr(args, "depth_peft_ckpt", "") or ""),
        "cad_registry": str(getattr(args, "cad_registry", "") or ""),
        "module_ablation": str(getattr(args, "module_ablation", "full")),
        "score_source_selected": str(getattr(args, "score_source", "fusion")),
        "fusion_rule": str(getattr(args, "fusion_rule", "sum")),
        "assist_fill_mode": str(getattr(args, "assist_fill", "train_mean")),
        "assist_stats_dir": str(getattr(args, "assist_stats_dir", "") or ""),
        "train_sample_ratio": float(getattr(args, "train_sample_ratio", 1.0)),
        "train_sample_num": resolve_train_sample_num(args),
        "sampling_mode": str(getattr(args, "sampling_mode", "uniform_time")),
        "train_sample_seed": int(getattr(args, "train_sample_seed", 42)),
        "train_root": str(args.train_root),
        "test_root": str(args.test_root),
        "status": "ok",
        "reason": "",
        "best_epoch": "N/A",
        "best_val_loss": None,
        "auroc": None,
        "num_patches": int(len(test_dataset)),
        "num_images": 0,
        "num_abnormal": 0,
        "num_normal": 0,
        "score_min": None,
        "score_max": None,
        "scores_csv": "",
        "scores_csv_by_source": {source: "" for source in SCORE_SOURCES},
        "auroc_by_source": {source: None for source in SCORE_SOURCES},
        "source_semantics": dict(SOURCE_SEMANTICS),
        "frame_ids": [],
        "output_log": str(args.output_log) if args.output_log else "",
    }

    if len(test_dataset) == 0:
        result["status"] = "skipped"
        result["reason"] = "no test samples for this view"
        print("\n" + "=" * 60)
        print(f"  Cam{args.view_id} Test Result")
        print("=" * 60)
        print("  AUROC: N/A (no test samples)")
        if args.output_log and args.append_log:
            append_eval_log(args.output_log, result)
            print(f"  appended to log: {args.output_log}")
        if args.result_json:
            Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.result_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # ---- 2. 构建模型并加载权重 ----
    teacher_rgb = ResNet50Encoder(pretrained=True).to(device).eval()
    teacher_depth = ResNet50Encoder(pretrained=True).to(device).eval()
    for p in teacher_rgb.parameters(): p.requires_grad = False
    for p in teacher_depth.parameters(): p.requires_grad = False

    module_ablation = str(getattr(args, "module_ablation", "full"))
    student_rgb = ResNet50DualModalDecoder(
        pretrained=False,
        module_ablation=module_ablation,
    ).to(device).eval()
    student_depth = ResNet50DualModalDecoder(
        pretrained=False,
        module_ablation=module_ablation,
    ).to(device).eval()

    if args.channels_last and device.type == "cuda":
        teacher_rgb = teacher_rgb.to(memory_format=torch.channels_last)
        teacher_depth = teacher_depth.to(memory_format=torch.channels_last)
        student_rgb = student_rgb.to(memory_format=torch.channels_last)
        student_depth = student_depth.to(memory_format=torch.channels_last)
        print("Memory format: channels_last")

    ckpt = safe_load_ckpt(args.ckpt, device)
    student_rgb.load_state_dict(strip_prefix(ckpt['student_rgb']))
    student_depth.load_state_dict(strip_prefix(ckpt['student_depth']))
    result["best_epoch"] = ckpt.get("epoch", "N/A")
    result["best_val_loss"] = ckpt.get("best_val_loss")
    best_val_loss_text = (
        f"{result['best_val_loss']:.4f}"
        if isinstance(result["best_val_loss"], (int, float))
        else "N/A"
    )
    print(f"Loaded epoch={result['best_epoch']}, best_val_loss={best_val_loss_text}")
    print(f"Module ablation: {module_ablation}")

    assist_feature_means = ensure_assist_feature_means(
        args,
        device,
        amp_dtype,
        teacher_rgb,
        teacher_depth,
    )

    explicit_depth_peft_ckpt = getattr(args, "depth_peft_ckpt", None)
    cad_registry = getattr(args, "cad_registry", None)
    joint_peft, joint_peft_meta = load_depth_peft_from_joint_ckpt(ckpt, device)
    routed_depth_peft_ckpt = None
    routed_meta = {}
    if explicit_depth_peft_ckpt:
        routed_depth_peft_ckpt = resolve_artifact_path(explicit_depth_peft_ckpt)
        routed_meta = {"source": "explicit_depth_peft_ckpt"}
    elif cad_registry:
        routed_depth_peft_ckpt, routed_meta = resolve_depth_peft_from_cad_registry(
            cad_registry,
            task_id=getattr(args, "task_id", None),
            view_id=int(args.view_id),
        )

    if routed_depth_peft_ckpt:
        peft, peft_meta = load_depth_peft(str(routed_depth_peft_ckpt), device)
        teacher_depth = DepthEncoderWithPEFT(teacher_depth, peft).to(device).eval()
        result["depth_peft_ckpt"] = str(routed_depth_peft_ckpt)
        result["depth_peft_gain"] = float(peft.gain.detach().cpu())
        result["depth_peft_bias"] = float(peft.bias.detach().cpu())
        result["depth_peft_meta"] = {**routed_meta, **peft_meta}
        print(
            "Loaded Depth PEFT: "
            f"{routed_depth_peft_ckpt} (gain={peft.gain.item():.6f}, bias={peft.bias.item():.6f})"
        )
    elif joint_peft is not None:
        teacher_depth = DepthEncoderWithPEFT(teacher_depth, joint_peft).to(device).eval()
        result["depth_peft_ckpt"] = f"{args.ckpt}::embedded"
        result["depth_peft_gain"] = float(joint_peft.gain.detach().cpu())
        result["depth_peft_bias"] = float(joint_peft.bias.detach().cpu())
        result["depth_peft_meta"] = joint_peft_meta
        print(
            "Loaded embedded Depth PEFT: "
            f"{args.ckpt} (gain={joint_peft.gain.item():.6f}, bias={joint_peft.bias.item():.6f})"
        )
    elif cad_registry:
        result["depth_peft_meta"] = routed_meta
        print(f"No CAD PEFT route matched Cam{args.view_id}; falling back to baseline teacher_depth.")

    # ---- 3. 评估 ----
    eval_payload = evaluate(
        teacher_rgb, teacher_depth, student_rgb, student_depth,
        test_loader, device,
        amp_dtype=amp_dtype,
        channels_last=args.channels_last,
        assist_fill=result["assist_fill_mode"],
        assist_feature_means=assist_feature_means,
        fusion_rule=result["fusion_rule"],
    )
    selected_source = result["score_source_selected"]
    labels = eval_payload["labels"]
    frame_ids = eval_payload["frame_ids"]
    scores_by_source = eval_payload["scores_by_source"]
    auroc_by_source = eval_payload["auroc_by_source"]

    scores = scores_by_source[selected_source]
    auroc = auroc_by_source[selected_source]

    result["auroc"] = float(auroc) if auroc is not None else None
    result["auroc_by_source"] = {
        source: (float(value) if value is not None else None)
        for source, value in auroc_by_source.items()
    }
    result["frame_ids"] = list(frame_ids)
    result["num_images"] = int(len(scores))
    result["num_abnormal"] = int((labels == 1).sum()) if len(labels) else 0
    result["num_normal"] = int((labels == 0).sum()) if len(labels) else 0
    if len(scores):
        result["score_min"] = float(scores.min())
        result["score_max"] = float(scores.max())

    print("\n" + "=" * 60)
    print(f"  Cam{args.view_id} Test Result")
    print("=" * 60)
    print(f"  Module ablation: {result['module_ablation']}")
    print(f"  Score source: {selected_source}")
    print(f"  Fusion rule: {result['fusion_rule']}")
    if auroc is None:
        print(f"  AUROC: N/A (single-class test set)")
    else:
        print(f"  AUROC: {auroc:.4f}")
    print(f"  AUROC by source: {format_metric_map(result['auroc_by_source'])}")
    print(f"  #images: {len(scores)}")
    if len(scores):
        print(f"  score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  #abnormal: {(labels == 1).sum() if len(labels) else 0}, "
          f"#normal: {(labels == 0).sum() if len(labels) else 0}")

    # 把分数落盘，便于后续画分布图
    if len(scores):
        if args.scores_dir:
            scores_dir = Path(args.scores_dir)
            scores_dir.mkdir(parents=True, exist_ok=True)
            for source in SCORE_SOURCES:
                source_path = scores_dir / f"scores_{source}.csv"
                write_scores_csv(source_path, frame_ids, scores_by_source[source], labels)
                result["scores_csv_by_source"][source] = str(source_path)

        out_csv = args.scores_csv
        if not out_csv:
            out_csv = (
                result["scores_csv_by_source"].get(selected_source)
                or default_scores_csv_path(args.ckpt, selected_source)
            )
        out_csv_path = Path(out_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        existing_selected = result["scores_csv_by_source"].get(selected_source, "")
        same_path = (
            existing_selected
            and out_csv_path.resolve(strict=False) == Path(existing_selected).resolve(strict=False)
        )
        if not same_path:
            write_scores_csv(out_csv_path, frame_ids, scores, labels)
        result["scores_csv"] = str(out_csv)
        if not result["scores_csv_by_source"].get(selected_source):
            result["scores_csv_by_source"][selected_source] = str(out_csv)
        print(f"  scores saved to: {out_csv}")

    if args.output_log and args.append_log:
        append_eval_log(args.output_log, result)
        print(f"  appended to log: {args.output_log}")

    if args.result_json:
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    evaluate_from_args(args)


if __name__ == "__main__":
    main()
