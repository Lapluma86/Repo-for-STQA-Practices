#!/usr/bin/env python3
"""
A2 test-set prediction script.

Loads the best-QWK checkpoint (standard GroupedModel, no MTL), mirrors the
training-time model construction from common/runner.py, enables auxiliary
attributes when the checkpoint was trained with them, applies validation
threshold calibration, and writes a result CSV.

Test layout (multi-package):
    <test_root>/<SCH_xxx>/test_hidden/<SCH_xxx>/<CLS_xxx>/<P_xxx>/audio|video/<feature>/<session>/sequence.npz
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.data import feature_io as _feature_io_mod
from common.data.dataset import (
    AUX_ATTR_COLS, FeatureConfig, ITEM_COLS, SESSIONS,
)
from common.data.grouped_dataset import (
    GroupedParticipantDataset, grouped_collate_fn,
)
from common.models.aux_encoder import AuxiliaryAttributeEncoder
from common.models.grouped_model import CORALHead, GroupedModel
from common.models.heads import A2OrdinalHead
from common.models.mtcn_backbone import BackboneConfig, MTCNBackbone
from common.runner import _decode_a2_logits, _normalize_decode_method

log = logging.getLogger("predict_a2")

DEFAULT_RUNS_DIR = Path("./output/runs")
DEFAULT_TEST_ROOT = Path("/data1/AdoDas/Test/test/test_hidden")
DEFAULT_OUTPUT_CSV = Path("./result_pred.csv")
TEMPLATE_CSV = Path("/data1/AdoDas/output/result.csv")


# ---------------------------------------------------------------------------
# Manifest discovery
# ---------------------------------------------------------------------------

PID_RE = re.compile(r"^P\d+$")
CLS_RE = re.compile(r"^CLS_\d+$")
SCH_RE = re.compile(r"^SCH_\d+$")


def discover_manifest(test_root: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Walk the test tree, return a manifest DataFrame and
    a mapping from school id to the root that contains its sessions.

    Each row is one (school, class, pid, session) — same shape as the training
    manifests consumed by GroupedParticipantDataset.
    """
    rows: list[dict[str, str]] = []
    school_to_root: dict[str, Path] = {}

    sch_dirs = sorted(p for p in test_root.iterdir() if p.is_dir() and SCH_RE.match(p.name))
    if not sch_dirs:
        raise FileNotFoundError(f"No SCH_xxx packages under {test_root}")

    log.info(f"Discovered {len(sch_dirs)} schools under {test_root}")
    for sch in sch_dirs:
        school_to_root[sch.name] = test_root

        cls_dirs = sorted(d for d in sch.iterdir() if d.is_dir() and CLS_RE.match(d.name))
        n_pid = 0
        n_sess = 0
        for cls in cls_dirs:
            pid_dirs = sorted(d for d in cls.iterdir() if d.is_dir() and PID_RE.match(d.name))
            for pid in pid_dirs:
                n_pid += 1
                probe_dirs = [
                    pid / "audio" / "mel_mfcc",
                    pid / "audio" / "vad",
                    pid / "video" / "face_behavior",
                ]
                sessions: set[str] = set()
                for probe in probe_dirs:
                    if probe.is_dir():
                        for s in probe.iterdir():
                            if s.is_dir() and s.name in SESSIONS:
                                sessions.add(s.name)
                        if sessions:
                            break
                for sess in sorted(sessions):
                    rows.append({
                        "anon_school": sch.name,
                        "anon_class": cls.name,
                        "anon_pid": pid.name,
                        "session": sess,
                    })
                    n_sess += 1
        log.info(f"  {sch.name}: {len(cls_dirs)} classes, {n_pid} participants, {n_sess} sessions")

    if not rows:
        raise RuntimeError(f"No sessions discovered under {test_root}")
    manifest = pd.DataFrame(rows)
    log.info(f"Total manifest rows: {len(manifest)} | unique participants: "
             f"{manifest[['anon_school','anon_class','anon_pid']].drop_duplicates().shape[0]}")
    return manifest, school_to_root


# ---------------------------------------------------------------------------
# Dataset: per-row package root switching
# ---------------------------------------------------------------------------

class TestPackageGroupedDataset(GroupedParticipantDataset):
    """GroupedParticipantDataset variant that swaps self.root per school
    before loading a session, so that the multi-package test layout works
    without changing the base class.

    Test labels do not exist — we set y_a1/y_a2 to -1 placeholders; the
    grouped dataset uses .get with default -1 which is fine. aux_attrs
    are also unknown → all -1, which the AuxiliaryAttributeEncoder maps
    to the "missing" embedding index 0.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        cfg: FeatureConfig,
        school_to_root: dict[str, Path],
    ) -> None:
        # Bypass GroupedParticipantDataset.__init__ to avoid reading a CSV path
        # and to inject our school->root mapping. Replicate the grouping logic.
        self.cfg = cfg
        self.split = ""  # path-effective; load_sequence ignores split,
                         # load_egemaps_pooled concatenates Path / "" → identity.
        self.session_drop_prob = 0.0
        self.school_to_root = school_to_root
        # self.root is set per-call by _load_single_session / _probe_dims
        # via _set_root_for_school(); seed it with any package.
        self.root = next(iter(school_to_root.values()))

        group_cols = ["anon_school", "anon_class", "anon_pid"]
        self.participants: list[dict[str, Any]] = []
        for (school, cls, pid), group in manifest.groupby(group_cols):
            sess_rows = {str(r["session"]): r for _, r in group.iterrows()}
            self.participants.append({
                "anon_school": str(school),
                "anon_class": str(cls),
                "anon_pid": str(pid),
                "sess_rows": sess_rows,
                # Test set has no labels — fill with -1 placeholders.
                "y_a1": np.full(3, -1.0, dtype=np.float32),
                "y_a2": np.full(len(ITEM_COLS), -1.0, dtype=np.float32),
                "aux_attrs": np.full(len(AUX_ATTR_COLS), -1.0, dtype=np.float32),
            })

        self._feature_dims: dict[str, int] | None = None
        self._cache: list[dict | None] | None = None

    def _set_root_for_school(self, school: str) -> None:
        root = self.school_to_root.get(str(school))
        if root is None:
            raise KeyError(f"No package root registered for school {school!r}")
        self.root = root

    def _load_single_session(self, row):
        self._set_root_for_school(row["anon_school"])
        return super()._load_single_session(row)

    def _probe_dims(self) -> dict[str, int]:
        # The base class probes via self.root; ensure root matches the first
        # participant's school before probing.
        first = self.participants[0]
        any_sess = next(iter(first["sess_rows"].values()))
        self._set_root_for_school(any_sess["anon_school"])
        return super()._probe_dims()


# ---------------------------------------------------------------------------
# Model construction (mirrors common/runner.py:main)
# ---------------------------------------------------------------------------

def build_model_from_config(cfg: dict, model_sd: dict[str, torch.Tensor]
                            ) -> tuple[GroupedModel, torch.nn.Module, FeatureConfig]:
    """Reconstruct the FeatureConfig, BackboneConfig, GroupedModel and
    participant_head exactly as in runner.main(), then return the modules
    ready to receive state_dict loads.
    """
    defaults = FeatureConfig()
    feat_cfg = FeatureConfig(
        feature_root=cfg.get("feature_root", defaults.feature_root),
        audio_features=list(cfg.get("audio_features", defaults.audio_features)),
        video_features=list(cfg.get("video_features", defaults.video_features)),
        audio_ssl_model_tag=cfg.get("audio_ssl_model_tag", defaults.audio_ssl_model_tag),
        video_ssl_model_tag=cfg.get("video_ssl_model_tag", defaults.video_ssl_model_tag),
        mask_policy=cfg.get("mask_policy", defaults.mask_policy),
        core_audio=list(cfg.get("core_audio", defaults.core_audio)),
        core_video=list(cfg.get("core_video", defaults.core_video)),
    )

    # Infer group dims from the checkpoint's adapter norm weights — this is
    # the source of truth and avoids re-probing the test set.
    audio_group_dims: dict[str, int] = {}
    audio_pooled_group_dims: dict[str, int] = {}
    video_group_dims: dict[str, int] = {}
    for k, v in model_sd.items():
        # sequence adapters: backbone.audio_adapters.<name>.norm.weight
        m = re.match(r"backbone\.(audio|video)_adapters\.([^.]+)\.norm\.weight$", k)
        if m:
            modality, name = m.group(1), m.group(2)
            if modality == "audio":
                audio_group_dims[name] = int(v.shape[0])
            else:
                video_group_dims[name] = int(v.shape[0])
            continue
        # pooled adapters: backbone.audio_pooled_adapters.<name>.0.weight (LayerNorm)
        m = re.match(r"backbone\.audio_pooled_adapters\.([^.]+)\.0\.weight$", k)
        if m:
            audio_pooled_group_dims[m.group(1)] = int(v.shape[0])

    log.info(f"Inferred audio seq dims: {audio_group_dims}")
    log.info(f"Inferred audio pooled dims: {audio_pooled_group_dims}")
    log.info(f"Inferred video dims: {video_group_dims}")

    bb_cfg = BackboneConfig(
        audio_group_dims=audio_group_dims,
        audio_pooled_group_dims=audio_pooled_group_dims,
        video_group_dims=video_group_dims,
        d_adapter=cfg.get("d_adapter", 64),
        d_model=cfg.get("d_model", 256),
        tcn_layers=cfg.get("tcn_layers", 6),
        tcn_kernel_size=cfg.get("tcn_kernel_size", 3),
        asp_alpha=cfg.get("asp_alpha", 0.5),
        asp_beta=cfg.get("asp_beta", 0.5),
        dropout=cfg.get("dropout", 0.2),
        d_shared=cfg.get("d_shared", 256),
    )
    backbone = MTCNBackbone(bb_cfg)

    aux_encoder = None
    aux_dim = 0
    if bool(cfg.get("use_aux_attrs", False)):
        aux_embed_dim = int(cfg.get("aux_embed_dim", 8))
        aux_encoder = AuxiliaryAttributeEncoder(
            embed_dim=aux_embed_dim, dropout=cfg.get("dropout", 0.2)
        )
        aux_dim = aux_encoder.output_dim
        log.info(f"Auxiliary attributes ENABLED: embed_dim={aux_embed_dim}, "
                 f"output_dim={aux_dim}")
    else:
        log.info("Auxiliary attributes DISABLED")

    grouped_model = GroupedModel(
        backbone=backbone,
        d_shared=bb_cfg.d_shared,
        aggregator_method=cfg.get("aggregator", "mlp"),
        dropout=cfg.get("dropout", 0.2),
        aux_encoder=aux_encoder,
    )

    head_in = bb_cfg.d_shared + aux_dim
    use_coral = bool(cfg.get("use_coral", False))
    if use_coral:
        participant_head = CORALHead(head_in)
        log.info(f"CORAL head built: in={head_in}, items=21, thresholds=3")
    else:
        participant_head = A2OrdinalHead(head_in)
        log.info(f"A2Ordinal head built: in={head_in}, items=21, thresholds=3")

    return grouped_model, participant_head, feat_cfg, audio_group_dims, video_group_dims


# ---------------------------------------------------------------------------
# SSL model tag auto-detection
# ---------------------------------------------------------------------------

def _match_ssl_tag(
    test_root: Path,
    feat_name: str,
    modality: str,
    expected_dim: int,
    preferred_tag: str,
) -> str:
    """Find an SSL model tag on disk whose feature dimension matches *expected_dim*.

    Scans the first available participant directory for the given feature name
    and tries each discovered model tag.  Falls back to *preferred_tag* if no
    match is found.
    """
    # Walk into the first participant we can find
    for sch in sorted(test_root.iterdir()):
        if not sch.is_dir():
            continue
        for cls in sorted(sch.iterdir()):
            if not cls.is_dir():
                continue
            for pid in sorted(cls.iterdir()):
                if not pid.is_dir():
                    continue
                feat_dir = pid / modality / feat_name
                if not feat_dir.is_dir():
                    continue
                # Try every model tag sub-directory
                for tag_dir in sorted(feat_dir.iterdir()):
                    if not tag_dir.is_dir():
                        continue
                    # Load first session sequence to get dim
                    sessions = sorted(
                        s for s in tag_dir.iterdir()
                        if s.is_dir() and s.name in SESSIONS
                    )
                    if not sessions:
                        continue
                    try:
                        arr = np.load(str(sessions[0] / "sequence.npz"))
                        probe_dim = arr["features"].shape[1]
                    except Exception:
                        continue
                    if probe_dim == expected_dim:
                        log.info(
                            "Auto-matched %s SSL: tag=%s dim=%d",
                            modality, tag_dir.name, probe_dim,
                        )
                        return tag_dir.name
                # If we reached here, no match in this participant — fall through
    log.warning(
        "Could not find %s SSL tag with dim=%d, using preferred %r",
        modality, expected_dim, preferred_tag,
    )
    return preferred_tag


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def load_calibration(run_dir: Path) -> tuple[str, np.ndarray | None]:
    cal_path = run_dir / "calibration" / "a2_threshold_offsets_grouped.json"
    if not cal_path.exists():
        log.warning(f"No calibration file at {cal_path}; using raw argmax decode")
        return "argmax", None
    with open(cal_path) as f:
        cal = json.load(f)
    strategy = cal.get("selected_strategy", "argmax")
    decode_method = cal.get("selected_decode_method", "argmax")
    log.info(f"Calibration strategy: {strategy} (decode={decode_method})")
    offsets = None
    if strategy.startswith("calibrated_"):
        strat_info = cal.get("strategies", {}).get(strategy, {})
        if "offsets" in strat_info:
            offsets = np.asarray(strat_info["offsets"], dtype=np.float32)
            log.info(f"Loaded calibrated offsets shape={offsets.shape}")
    return decode_method, offsets


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


@torch.no_grad()
def run_inference(
    grouped_model: GroupedModel,
    participant_head: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    decode_method: str,
    offsets: np.ndarray | None,
) -> list[dict[str, Any]]:
    grouped_model.eval()
    participant_head.eval()

    offsets_t = None
    if offsets is not None:
        offsets_t = torch.from_numpy(offsets).float().to(device)

    method = _normalize_decode_method(decode_method)
    log.info(f"Decoding logits with method='{method}'"
             + (" + threshold offsets" if offsets_t is not None else ""))

    out_rows: list[dict[str, Any]] = []
    pbar = tqdm(loader, desc="Predict", dynamic_ncols=True, unit="batch")
    for batch in pbar:
        if batch is None:
            continue
        flat = _to_device(batch["flat_batch"], device)
        valid = batch["session_valid"].to(device)
        aux = batch.get("participant_aux_attrs")
        aux = aux.to(device) if aux is not None else None
        n_p = int(batch["n_participants"])

        result = grouped_model(flat, n_p, valid, aux)
        logits = participant_head(result["participant_repr"]).float()
        if offsets_t is not None:
            logits = logits + offsets_t  # broadcast (n_items, n_thresholds)

        preds = _decode_a2_logits(participant_head, logits, decode_method=method)
        preds_np = preds.detach().cpu().numpy().astype(int)

        for i in range(n_p):
            row = {
                "anon_school": batch["anon_schools"][i],
                "anon_class": batch["anon_classes"][i],
                "anon_pid": batch["anon_pids"][i],
            }
            for j, col in enumerate(ITEM_COLS):
                row[col] = int(preds_np[i, j])
            out_rows.append(row)

        pbar.set_postfix({"participants": len(out_rows)})

    return out_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_latest_run(runs_dir: Path) -> Path | None:
    """Return the newest run directory under *runs_dir* that has checkpoints/best.pt."""
    if not runs_dir.is_dir():
        return None
    dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for d in dirs:
        if (d / "checkpoints" / "best.pt").exists():
            return d
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A2 test-set prediction")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Training run directory (auto-detected from ./output/runs/ if omitted)")
    p.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--template-csv", type=Path, default=TEMPLATE_CSV,
                   help="CSV with the canonical row order / columns to match")
    return p.parse_args()


def select_device(name: str) -> torch.device:
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    args = parse_args()

    # Auto-discover latest run if --run-dir not specified
    run_dir = args.run_dir
    if run_dir is None:
        run_dir = _find_latest_run(DEFAULT_RUNS_DIR)
        if run_dir is None:
            log.error("No run directory specified and no runs found under %s", DEFAULT_RUNS_DIR)
            return 1
        log.info("Auto-detected latest run: %s", run_dir)

    log.info("=" * 80)
    log.info("A2 prediction (best QWK checkpoint)")
    log.info("=" * 80)
    log.info(f"Run dir   : {run_dir}")
    log.info(f"Test root : {args.test_root}")
    log.info(f"Output csv: {args.output}")

    meta_path = run_dir / "run_meta.json"
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    cfg_path = run_dir / "config_used.yaml"
    for p in (meta_path, ckpt_path):
        if not p.exists():
            log.error(f"Missing required file: {p}")
            return 1

    meta = json.load(open(meta_path))
    cfg = meta.get("full_config") or {}
    if not cfg and cfg_path.exists():
        cfg = yaml.safe_load(open(cfg_path)) or {}
    log.info(f"Best epoch: {meta.get('best_epoch')} | "
             f"final selected QWK: {meta.get('final_selected_metrics', {}).get('mean_qwk')}")
    log.info(f"Audio SSL : {cfg.get('audio_ssl_model_tag')} | "
             f"Video SSL : {cfg.get('video_ssl_model_tag')}")
    log.info(f"use_coral={cfg.get('use_coral')} | "
             f"use_aux_attrs={cfg.get('use_aux_attrs')} | "
             f"aggregator={cfg.get('aggregator')}")

    device = select_device(args.device)
    log.info(f"Device: {device}")

    log.info("Loading checkpoint ...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    full_sd = ckpt["model_state_dict"]

    # Extract grouped_model.* and participant_head.* from the flat state dict.
    model_sd: dict[str, torch.Tensor] = {}
    head_sd: dict[str, torch.Tensor] = {}
    for k, v in full_sd.items():
        if k.startswith("grouped_model."):
            model_sd[k[len("grouped_model."):]] = v
        elif k.startswith("participant_head."):
            head_sd[k[len("participant_head."):]] = v

    if not model_sd:
        log.error("No grouped_model.* keys found in checkpoint")
        return 2
    if not head_sd:
        log.error("No participant_head.* keys found in checkpoint")
        return 2

    log.info(f"Checkpoint epoch={ckpt.get('epoch')} best_metric={ckpt.get('best_metric'):.4f}")
    log.info(f"Extracted {len(model_sd)} grouped_model keys + {len(head_sd)} participant_head keys")
    if ckpt.get("enable_mtl"):
        log.info("MTL checkpoint detected — non-grouped_model keys (aux_task_head, "
                 "session_head, etc.) are skipped at load time")

    grouped_model, participant_head, feat_cfg, audio_group_dims, video_group_dims = build_model_from_config(cfg, model_sd)

    missing, unexpected = grouped_model.load_state_dict(model_sd, strict=False)
    if missing:
        log.warning(f"GroupedModel missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        log.warning(f"GroupedModel unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    participant_head.load_state_dict(head_sd, strict=True)
    grouped_model.to(device).eval()
    participant_head.to(device).eval()

    n_params = sum(p.numel() for p in grouped_model.parameters()) \
        + sum(p.numel() for p in participant_head.parameters())
    log.info(f"Model loaded ({n_params/1e6:.2f}M params)")

    # Auto-match SSL model tags to the dimensions the checkpoint expects.
    # This prevents 768-vs-1024 dimension mismatches when the HDF5-packed
    # features used a different SSL model than the one written in config_used.yaml.
    expected_audio_ssl_dim = audio_group_dims.get("ssl_embed")
    expected_video_ssl_dim = video_group_dims.get("vision_ssl_embed")
    for feat_name, tag_attr, expected_dim in [
        ("ssl_embed", "audio_ssl_model_tag", expected_audio_ssl_dim),
        ("vision_ssl_embed", "video_ssl_model_tag", expected_video_ssl_dim),
    ]:
        if expected_dim is None:
            continue
        preferred = cfg.get(tag_attr, "")
        matched = _match_ssl_tag(
            args.test_root, feat_name,
            "audio" if "audio" in tag_attr else "video",
            expected_dim, str(preferred),
        )
        if matched != preferred:
            setattr(feat_cfg, tag_attr, matched)
            log.info("Overriding %s: %s -> %s", tag_attr, preferred, matched)

    log.info("Discovering test manifest ...")
    manifest, school_to_root = discover_manifest(args.test_root)
    n_unique_p = manifest[["anon_school", "anon_class", "anon_pid"]].drop_duplicates().shape[0]
    log.info(f"Manifest ready: {len(manifest)} session rows, {n_unique_p} participants")

    log.info("Building dataset ...")
    dataset = TestPackageGroupedDataset(manifest, feat_cfg, school_to_root)
    log.info(f"Probing feature dims from first participant ...")
    dims = dataset.feature_dims
    log.info(f"Probed dims: {dims}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=grouped_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    decode_method, offsets = load_calibration(run_dir)

    log.info(f"Running inference: {len(dataset)} participants, "
             f"batch_size={args.batch_size}, workers={args.num_workers} ...")
    rows = run_inference(grouped_model, participant_head, loader, device,
                         decode_method, offsets)
    log.info(f"Inference complete: {len(rows)} predictions")

    pred_df = pd.DataFrame(rows)

    # Order rows to match the template CSV.
    if args.template_csv.exists():
        tmpl = pd.read_csv(args.template_csv)
        key_cols = ["anon_school", "anon_class", "anon_pid"]
        tmpl_keys = tmpl[key_cols].astype(str)
        pred_df[key_cols] = pred_df[key_cols].astype(str)
        merged = tmpl_keys.merge(pred_df, on=key_cols, how="left", indicator=True)
        missing_keys = (merged["_merge"] == "left_only").sum()
        extra = pred_df.merge(tmpl_keys, on=key_cols, how="left", indicator=True)
        extra_keys = (extra["_merge"] == "left_only").sum()
        log.info(f"Template alignment: {len(tmpl)} rows | "
                 f"missing in preds: {missing_keys} | extra in preds: {extra_keys}")
        merged = merged.drop(columns=["_merge"])
        # Ensure column order matches template
        cols = list(tmpl.columns)
        merged = merged[cols]
        out_df = merged
    else:
        cols = ["anon_school", "anon_class", "anon_pid", *ITEM_COLS]
        out_df = pred_df[cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    log.info(f"Wrote {len(out_df)} rows -> {args.output}")
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
