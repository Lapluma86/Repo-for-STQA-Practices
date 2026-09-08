#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared protocol helpers for unified RGB and RGB+Depth baseline experiments."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from datasets.rail_dataset import RailDualModalDataset


PROJ_ROOT = Path(__file__).resolve().parents[2]
SERVER_TRAIN_ROOT = "/data1/Leaddo_data/20260327-resize512"
SERVER_TEST_ROOT = "/home/root123/LF/WYM/SteelRailWay/rail_mvtec_gt_test"
DEFAULT_IMG_SIZE = 512
DEFAULT_PATCH_SIZE = 900
DEFAULT_PATCH_STRIDE = 850
DEFAULT_SAMPLE_NUM = 1500
DEFAULT_SPLIT = (0.9, 0.1, 0.0)
DEFAULT_SAMPLING_MODE = "uniform_time"
FUSION_SOURCES = ("rgb", "depth", "fusion")


@dataclass
class ProtocolConfig:
    method: str
    train_root: str = SERVER_TRAIN_ROOT
    test_root: str = SERVER_TEST_ROOT
    view_id: int = 4
    img_size: int = DEFAULT_IMG_SIZE
    patch_size: int = DEFAULT_PATCH_SIZE
    patch_stride: int = DEFAULT_PATCH_STRIDE
    train_sample_num: int = DEFAULT_SAMPLE_NUM
    train_val_test_split: tuple[float, float, float] = DEFAULT_SPLIT
    sampling_mode: str = DEFAULT_SAMPLING_MODE
    device: str = "cuda:0"
    precision: str = "bf16"
    seed: int = 42
    batch_size: int = 32
    eval_batch_size: int = 8
    num_workers: int = 0
    preload: bool = False
    preload_workers: int = 0
    smoke: bool = False
    smoke_train_images: int = 16
    smoke_train_img_size: int = 256
    save_ckpt: bool = False
    save_maps: bool = False
    export_bank: bool = False
    output_root: str = ""
    results_root: str = ""
    result_json_path: str = ""
    scores_csv_path: str = ""
    input_mode: str = "rgb"
    depth_norm: str = "zscore"

    def output_dir(self) -> Path:
        if self.output_root:
            return Path(self.output_root)
        prefix = "baselines_rgbd" if self.input_mode == "rgbd" else "baselines_rgb"
        smoke_prefix = f"{prefix}_smoke"
        if self.smoke:
            return PROJ_ROOT / "$out" / smoke_prefix / f"cam{self.view_id}" / self.method
        return PROJ_ROOT / "outputs" / prefix / f"cam{self.view_id}" / self.method

    def result_dir(self) -> Path:
        if self.results_root:
            return Path(self.results_root)
        prefix = "baselines_rgbd" if self.input_mode == "rgbd" else "baselines_rgb"
        smoke_prefix = f"{prefix}_smoke"
        if self.smoke:
            return PROJ_ROOT / "$out" / smoke_prefix / f"cam{self.view_id}" / self.method
        return PROJ_ROOT / "results" / prefix / f"cam{self.view_id}" / self.method


def resolve_train_root(path: str) -> str:
    candidate = Path(path)
    if candidate.exists():
        return str(candidate.resolve())
    local_train = PROJ_ROOT / "data_20260327"
    if local_train.exists():
        return str(local_train.resolve())
    return str(path)


def resolve_test_root(path: str) -> str:
    candidate = Path(path)
    if candidate.exists():
        return str(candidate.resolve())
    local_test = PROJ_ROOT / "rail_mvtec_gt_test"
    if local_test.exists():
        return str(local_test.resolve())
    return str(path)


def _default_gt(base: RailDualModalDataset) -> torch.Tensor:
    return torch.zeros((base.img_size, base.img_size), dtype=torch.float32)


class RGBOnlyRailDataset(Dataset):
    """Wrap RailDualModalDataset and expose RGB-only samples."""

    def __init__(self, base: RailDualModalDataset):
        self.base = base
        self.samples = base.samples
        self.num_patches = getattr(base, "num_patches", 1)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        return {
            "image": item["intensity"],
            "label": int(item["label"]),
            "gt": item.get("gt", _default_gt(self.base)),
            "frame_id": item.get("frame_id", ""),
            "patch_idx": int(item.get("patch_idx", 0)),
            "view_id": int(item.get("view_id", self.base.view_id)),
        }


class RGBDRailDataset(Dataset):
    """Wrap RailDualModalDataset and expose paired RGB+Depth samples."""

    def __init__(self, base: RailDualModalDataset):
        self.base = base
        self.samples = base.samples
        self.num_patches = getattr(base, "num_patches", 1)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        image = item["intensity"]
        return {
            "image": image,
            "intensity": image,
            "rgb": image,
            "depth": item["depth"],
            "label": int(item["label"]),
            "gt": item.get("gt", _default_gt(self.base)),
            "frame_id": item.get("frame_id", ""),
            "patch_idx": int(item.get("patch_idx", 0)),
            "view_id": int(item.get("view_id", self.base.view_id)),
        }


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def protocol_log(config: ProtocolConfig, train_count: int, val_count: int) -> None:
    print(f"[Protocol] method={config.method}")
    print(f"[Protocol] input_mode={config.input_mode}")
    print(f"[Protocol] train_root={config.train_root}")
    print(f"[Protocol] test_root={config.test_root}")
    print(
        f"[Protocol] sample plan: {config.train_sample_num} -> "
        f"{train_count} train / {val_count} val"
    )
    print(f"[Protocol] img_size={config.img_size}")
    print(f"[Protocol] sampling_mode={config.sampling_mode}")
    print(f"[Protocol] depth_norm={config.depth_norm}")
    print(f"[Protocol] test patch={config.patch_size}/{config.patch_stride}")
    if config.smoke:
        print(f"[Protocol] smoke_train_img_size={config.smoke_train_img_size}")
        print("[Protocol] smoke mode enabled")


def dataset_summary(dataset: RailDualModalDataset | RGBOnlyRailDataset | RGBDRailDataset) -> dict[str, int]:
    samples = dataset.samples if hasattr(dataset, "samples") else []
    num_images = len(samples)
    num_good = sum(int(sample["label"]) == 0 for sample in samples)
    num_broken = sum(int(sample["label"]) == 1 for sample in samples)
    num_patches = getattr(dataset, "num_patches", 1)
    return {
        "num_images": int(num_images),
        "num_good_images": int(num_good),
        "num_broken_images": int(num_broken),
        "num_patch_samples": int(len(dataset)),
        "num_patches_per_image": int(num_patches),
    }


def _wrap_dataset(base: RailDualModalDataset, input_mode: str) -> RGBOnlyRailDataset | RGBDRailDataset:
    if input_mode == "rgbd":
        return RGBDRailDataset(base)
    return RGBOnlyRailDataset(base)


def _build_train_dataset(config: ProtocolConfig, split: str) -> RailDualModalDataset:
    train_img_size = config.smoke_train_img_size if config.smoke else config.img_size
    return RailDualModalDataset(
        train_root=config.train_root,
        test_root=config.test_root,
        view_id=config.view_id,
        split=split,
        img_size=train_img_size,
        depth_norm=config.depth_norm,
        use_patch=False,
        train_sample_num=config.train_sample_num,
        random_seed=config.seed,
        preload=config.preload,
        preload_workers=config.preload_workers,
        train_val_test_split=list(config.train_val_test_split),
        sampling_mode=config.sampling_mode,
    )


def _build_test_dataset(
    config: ProtocolConfig,
    *,
    img_size: int | None = None,
    use_patch: bool = True,
) -> RailDualModalDataset:
    return RailDualModalDataset(
        train_root=config.train_root,
        test_root=config.test_root,
        view_id=config.view_id,
        split="test",
        img_size=config.img_size if img_size is None else int(img_size),
        depth_norm=config.depth_norm,
        use_patch=use_patch,
        patch_size=config.patch_size,
        patch_stride=config.patch_stride,
        preload=config.preload,
        preload_workers=config.preload_workers,
    )


def _select_smoke_samples(samples: list[dict], wanted: int, val_ratio: float) -> tuple[list[dict], list[dict]]:
    good_samples = [sample for sample in samples if int(sample["label"]) == 0]
    if not good_samples:
        raise RuntimeError("Smoke fallback requires at least one good sample in test set")
    take = min(len(good_samples), max(2, wanted))
    chosen = list(good_samples[:take])
    budget_val = max(1, int(round(len(chosen) * val_ratio))) if len(chosen) > 1 else 0
    if budget_val >= len(chosen):
        budget_val = max(1, len(chosen) - 1)
    stride = max(2, int(round(1.0 / max(val_ratio, 1e-6))))
    val_samples = chosen[::stride] if budget_val > 0 else []
    val_samples = val_samples[:budget_val]
    val_ids = {sample["frame_id"] for sample in val_samples}
    train_samples = [sample for sample in chosen if sample["frame_id"] not in val_ids]
    if not train_samples:
        train_samples = chosen[:-1]
        val_samples = chosen[-1:]
    return train_samples, val_samples


def _select_smoke_test_samples(samples: list[dict], wanted: int) -> list[dict]:
    good = [sample for sample in samples if int(sample["label"]) == 0]
    broken = [sample for sample in samples if int(sample["label"]) == 1]
    half = max(1, wanted // 2)
    chosen = list(good[:half])
    if broken:
        chosen.extend(broken[: max(1, wanted - len(chosen))])
    if len(chosen) < wanted:
        remaining_ids = {sample["frame_id"] for sample in chosen}
        pool = [sample for sample in samples if sample["frame_id"] not in remaining_ids]
        chosen.extend(pool[: max(0, wanted - len(chosen))])
    return chosen


def _build_smoke_fallback(
    config: ProtocolConfig,
) -> tuple[RGBOnlyRailDataset | RGBDRailDataset, RGBOnlyRailDataset | RGBDRailDataset]:
    base_test = _build_test_dataset(config)
    train_samples, val_samples = _select_smoke_samples(
        base_test.samples,
        wanted=config.smoke_train_images,
        val_ratio=float(config.train_val_test_split[1]),
    )
    smoke_img_size = config.smoke_train_img_size if config.smoke else config.img_size
    train_base = _build_test_dataset(config, img_size=smoke_img_size, use_patch=False)
    val_base = _build_test_dataset(config, img_size=smoke_img_size, use_patch=False)
    train_base.samples = list(train_samples)
    val_base.samples = list(val_samples)
    train_base.num_patches = 1
    val_base.num_patches = 1
    return _wrap_dataset(train_base, config.input_mode), _wrap_dataset(val_base, config.input_mode)


def build_protocol_datasets(
    config: ProtocolConfig,
) -> dict[str, RGBOnlyRailDataset | RGBDRailDataset]:
    config.train_root = resolve_train_root(config.train_root)
    config.test_root = resolve_test_root(config.test_root)
    train_exists = Path(config.train_root).exists()
    if not train_exists and not config.smoke:
        raise FileNotFoundError(
            f"train_root does not exist: {config.train_root}. "
            "Use --smoke for local fallback without a training dataset."
        )

    if train_exists:
        train_base = _build_train_dataset(config, "train")
        val_base = _build_train_dataset(config, "val")
        train_ds = _wrap_dataset(train_base, config.input_mode)
        val_ds = _wrap_dataset(val_base, config.input_mode)
    else:
        train_ds, val_ds = _build_smoke_fallback(config)

    test_base = _build_test_dataset(config)
    test_ds = _wrap_dataset(test_base, config.input_mode)
    if config.smoke and hasattr(test_ds, "base") and hasattr(test_ds.base, "samples"):
        limit = min(len(test_ds.base.samples), max(6, config.smoke_train_images))
        test_ds.base.samples = _select_smoke_test_samples(list(test_ds.base.samples), limit)
        test_ds.samples = test_ds.base.samples
        test_ds.base.use_patch = False
        test_ds.base.num_patches = 1
        test_ds.num_patches = 1
    protocol_log(config, len(train_ds.samples), len(val_ds.samples))
    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
    }


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool = False,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def compute_safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def compute_pixel_stats(pixel_labels: np.ndarray) -> dict[str, int]:
    flat = np.asarray(pixel_labels).reshape(-1)
    positives = int(np.count_nonzero(flat > 0))
    total = int(flat.size)
    return {
        "total": total,
        "positive": positives,
        "negative": int(total - positives),
    }


def aggregate_patch_scores(
    frame_ids: Iterable[str],
    labels: Iterable[int],
    patch_maps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    frame_scores: dict[str, list[float]] = {}
    frame_labels: dict[str, int] = {}
    pixel_scores = patch_maps.reshape(patch_maps.shape[0], -1)
    labels_np = np.asarray(list(labels), dtype=np.int64)
    for idx, frame_id in enumerate(frame_ids):
        frame_scores.setdefault(frame_id, []).append(float(patch_maps[idx].max()))
        frame_labels[frame_id] = int(labels_np[idx])

    ordered_frame_ids = sorted(frame_scores.keys())
    image_scores = np.array([max(frame_scores[fid]) for fid in ordered_frame_ids], dtype=np.float64)
    image_labels = np.array([frame_labels[fid] for fid in ordered_frame_ids], dtype=np.int64)
    return image_scores, image_labels, ordered_frame_ids, pixel_scores.reshape(-1), image_labels


def reduce_patch_predictions(
    frame_scores: dict[str, list[float]],
    frame_labels: dict[str, int],
    patch_maps: list[np.ndarray],
    patch_gts: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    ordered_frame_ids = sorted(frame_scores.keys())
    image_scores = np.array([max(frame_scores[fid]) for fid in ordered_frame_ids], dtype=np.float64)
    image_labels = np.array([frame_labels[fid] for fid in ordered_frame_ids], dtype=np.int64)
    pixel_scores = np.concatenate(patch_maps, axis=0).reshape(-1)
    pixel_labels = np.concatenate(patch_gts, axis=0).reshape(-1)
    return image_scores, image_labels, ordered_frame_ids, pixel_scores, pixel_labels


def _frame_aggregate_from_patch_scores(
    patch_frame_ids: Sequence[str],
    patch_labels: Sequence[int],
    patch_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame_scores: dict[str, list[float]] = {}
    frame_labels: dict[str, int] = {}
    for idx, frame_id in enumerate(patch_frame_ids):
        frame_scores.setdefault(frame_id, []).append(float(patch_scores[idx]))
        frame_labels[frame_id] = int(patch_labels[idx])
    ordered_frame_ids = sorted(frame_scores.keys())
    image_scores = np.array([max(frame_scores[fid]) for fid in ordered_frame_ids], dtype=np.float64)
    image_labels = np.array([frame_labels[fid] for fid in ordered_frame_ids], dtype=np.int64)
    return image_scores, image_labels, ordered_frame_ids


def _compute_fusion_maps(
    rgb_maps: np.ndarray,
    depth_maps: np.ndarray,
    fusion_rule: str,
) -> tuple[np.ndarray, dict]:
    if fusion_rule == "sum":
        return (rgb_maps + depth_maps).astype(np.float32, copy=False), {
            "fusion_rule": "sum",
        }
    if fusion_rule != "max_norm":
        raise ValueError(f"Unsupported fusion_rule: {fusion_rule}")
    rgb_min = float(np.min(rgb_maps))
    rgb_max = float(np.max(rgb_maps))
    depth_min = float(np.min(depth_maps))
    depth_max = float(np.max(depth_maps))
    rgb_denom = max(rgb_max - rgb_min, 1e-8)
    depth_denom = max(depth_max - depth_min, 1e-8)
    rgb_norm = (rgb_maps - rgb_min) / rgb_denom if rgb_max > rgb_min else np.zeros_like(rgb_maps, dtype=np.float32)
    depth_norm = (
        (depth_maps - depth_min) / depth_denom
        if depth_max > depth_min
        else np.zeros_like(depth_maps, dtype=np.float32)
    )
    fusion = np.maximum(rgb_norm, depth_norm).astype(np.float32, copy=False)
    return fusion, {
        "fusion_rule": "max_norm",
        "rgb_global_min": rgb_min,
        "rgb_global_max": rgb_max,
        "depth_global_min": depth_min,
        "depth_global_max": depth_max,
    }


def reduce_multisource_patch_predictions(
    patch_frame_ids: Sequence[str],
    patch_labels: Sequence[int],
    patch_score_chunks_by_source: dict[str, list[np.ndarray]],
    patch_gt_chunks: list[np.ndarray],
    *,
    fusion_rule: str = "sum",
    selected_source: str = "fusion",
) -> dict:
    if not patch_score_chunks_by_source:
        raise RuntimeError("No patch predictions were collected")
    flat_maps_by_source = {
        source: np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        for source, chunks in patch_score_chunks_by_source.items()
    }
    fusion_meta = {"fusion_rule": fusion_rule}
    if "fusion" not in flat_maps_by_source:
        if "rgb" not in flat_maps_by_source or "depth" not in flat_maps_by_source:
            raise KeyError("Fusion requires both rgb and depth patch maps")
        fusion_maps, fusion_meta = _compute_fusion_maps(
            flat_maps_by_source["rgb"],
            flat_maps_by_source["depth"],
            fusion_rule,
        )
        flat_maps_by_source["fusion"] = fusion_maps

    source_order = [source for source in FUSION_SOURCES if source in flat_maps_by_source]
    pixel_labels = np.concatenate(patch_gt_chunks, axis=0).reshape(-1).astype(np.float32, copy=False)
    image_scores_by_source: dict[str, np.ndarray] = {}
    pixel_scores_by_source: dict[str, np.ndarray] = {}
    image_labels: np.ndarray | None = None
    frame_ids: list[str] | None = None

    for source in source_order:
        flat_maps = flat_maps_by_source[source]
        pixel_scores_by_source[source] = flat_maps.reshape(-1).astype(np.float32, copy=False)
        patch_scores = flat_maps.reshape(flat_maps.shape[0], -1).max(axis=1).astype(np.float64, copy=False)
        source_image_scores, source_image_labels, source_frame_ids = _frame_aggregate_from_patch_scores(
            patch_frame_ids,
            patch_labels,
            patch_scores,
        )
        image_scores_by_source[source] = source_image_scores
        if image_labels is None:
            image_labels = source_image_labels
        if frame_ids is None:
            frame_ids = source_frame_ids

    if selected_source not in image_scores_by_source:
        raise KeyError(f"selected_source={selected_source} not available: {list(image_scores_by_source)}")
    assert image_labels is not None
    assert frame_ids is not None
    return {
        "frame_ids": frame_ids,
        "image_labels": image_labels,
        "image_scores": image_scores_by_source[selected_source],
        "pixel_scores": pixel_scores_by_source[selected_source],
        "image_scores_by_source": image_scores_by_source,
        "pixel_scores_by_source": pixel_scores_by_source,
        "pixel_labels": pixel_labels,
        "pixel_stats": compute_pixel_stats(pixel_labels),
        "fusion_meta": fusion_meta,
        "source_order": source_order,
    }


def compute_source_aurocs(
    image_labels: np.ndarray,
    image_scores_by_source: dict[str, np.ndarray],
    pixel_labels: np.ndarray,
    pixel_scores_by_source: dict[str, np.ndarray],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    image_auroc_by_source = {
        source: compute_safe_auroc(image_labels, scores)
        for source, scores in image_scores_by_source.items()
    }
    pixel_labels_int = pixel_labels.astype(np.int64, copy=False)
    pixel_auroc_by_source = {
        source: compute_safe_auroc(pixel_labels_int, scores)
        for source, scores in pixel_scores_by_source.items()
    }
    return image_auroc_by_source, pixel_auroc_by_source


def write_scores_csv(path: Path, frame_ids: list[str], labels: np.ndarray, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-scores)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "frame_id", "label", "score"])
        writer.writeheader()
        for rank, idx in enumerate(order):
            writer.writerow(
                {
                    "rank": int(rank),
                    "frame_id": frame_ids[int(idx)],
                    "label": int(labels[int(idx)]),
                    "score": f"{float(scores[int(idx)]):.8f}",
                }
            )


def write_result_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def default_result_paths(config: ProtocolConfig) -> tuple[Path, Path]:
    if config.result_json_path:
        result_json = Path(config.result_json_path)
        result_json.parent.mkdir(parents=True, exist_ok=True)
    else:
        result_dir = config.result_dir()
        result_dir.mkdir(parents=True, exist_ok=True)
        result_json = result_dir / "result.json"
    if config.scores_csv_path:
        scores_csv = Path(config.scores_csv_path)
        scores_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        scores_csv = result_json.with_name("scores.csv")
    return result_json, scores_csv


def default_scores_csv_by_source(
    config: ProtocolConfig,
    sources: Sequence[str],
    *,
    selected_source: str = "fusion",
) -> tuple[Path, dict[str, Path]]:
    result_json, selected_scores = default_result_paths(config)
    if config.scores_csv_path:
        base_path = Path(config.scores_csv_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        stem = base_path.stem
        suffix = base_path.suffix or ".csv"
        if stem.endswith(f"_{selected_source}"):
            prefix = stem[: -len(f"_{selected_source}")]
        else:
            prefix = stem
        scores_by_source = {}
        for source in sources:
            if source == selected_source:
                scores_by_source[source] = base_path
            else:
                scores_by_source[source] = base_path.with_name(f"{prefix}_{source}{suffix}")
    else:
        scores_by_source = {
            source: result_json.with_name(f"scores_{source}.csv")
            for source in sources
        }
        selected_scores = scores_by_source[selected_source]
    return result_json, scores_by_source


def make_method_result(
    *,
    config: ProtocolConfig,
    image_auroc: float | None,
    pixel_auroc: float | None,
    train_summary: dict,
    val_summary: dict,
    test_summary: dict,
    scores_csv: Path,
    extra: dict | None = None,
) -> dict:
    payload = {
        "method": config.method,
        "cam": f"cam{config.view_id}",
        "view_id": int(config.view_id),
        "input_mode": str(config.input_mode),
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "img_size": int(config.img_size),
        "patch_size": int(config.patch_size),
        "patch_stride": int(config.patch_stride),
        "train_root": str(config.train_root),
        "test_root": str(config.test_root),
        "train_sample_num": int(config.train_sample_num),
        "train_val_test_split": [float(x) for x in config.train_val_test_split],
        "sampling_mode": str(config.sampling_mode),
        "depth_norm": str(config.depth_norm),
        "smoke": bool(config.smoke),
        "train_summary": train_summary,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "scores_csv": str(scores_csv),
    }
    if extra:
        payload.update(extra)
    return payload


def write_summary_files(summary_rows: list[dict], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    base_fieldnames = [
        "method",
        "view_id",
        "image_auroc",
        "pixel_auroc",
        "smoke",
        "train_images",
        "val_images",
        "test_images",
        "result_json",
    ]
    extra_fieldnames: list[str] = []
    for row in summary_rows:
        for key in row.keys():
            if key not in base_fieldnames and key not in extra_fieldnames:
                extra_fieldnames.append(key)
    fieldnames = base_fieldnames + extra_fieldnames
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    write_result_json(json_path, {"rows": summary_rows})


def precision_to_dtype(precision: str) -> torch.dtype | None:
    precision = precision.lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def ensure_parent(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
