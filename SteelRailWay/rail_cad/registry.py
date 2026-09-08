"""Registry helpers for append-only task-aware PEFT routing."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
REGISTRY_FILENAME = "registry.json"
ACTIVE_MAP_FILENAME = "active_depth_peft_map.json"


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_task_id(task_id) -> str:
    if task_id is None:
        raise ValueError("task_id is required")
    text = str(task_id).strip()
    if not text:
        raise ValueError("task_id must not be empty")
    try:
        return str(int(text))
    except ValueError:
        return text


def task_id_to_view_id(task_id, view_id: int | None = None) -> int:
    if view_id is not None:
        return int(view_id)
    normalized = normalize_task_id(task_id)
    try:
        return int(normalized)
    except ValueError as exc:
        raise ValueError(f"L1 CAD requires task_id to map to an integer view_id, got: {task_id}") from exc


def resolve_artifact_path(value) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return PROJECT_ROOT / path


def to_registry_path_text(value) -> str:
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_registry_root(path_or_root) -> Path:
    path = Path(path_or_root)
    if path.name == REGISTRY_FILENAME:
        return path.parent
    return path


def resolve_registry_path(path_or_root) -> Path:
    root = resolve_registry_root(path_or_root)
    return root / REGISTRY_FILENAME


def registry_snapshot_dir(path_or_root) -> Path:
    return resolve_registry_root(path_or_root) / "snapshots"


def registry_steps_dir(path_or_root) -> Path:
    return resolve_registry_root(path_or_root) / "steps"


def ensure_registry_dirs(path_or_root) -> None:
    root = resolve_registry_root(path_or_root)
    root.mkdir(parents=True, exist_ok=True)
    registry_snapshot_dir(root).mkdir(parents=True, exist_ok=True)
    registry_steps_dir(root).mkdir(parents=True, exist_ok=True)


def default_registry(sequence_name: str, defaults: dict | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id_type": "view_id",
        "sequence_name": str(sequence_name),
        "created_at": _now_text(),
        "task_order": [],
        "tasks": {},
        "steps": [],
        "defaults": dict(defaults or {}),
    }


def load_registry(path_or_root) -> dict:
    path = resolve_registry_path(path_or_root)
    if not path.exists():
        raise FileNotFoundError(f"CAD registry not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    registry.setdefault("task_order", [])
    registry.setdefault("tasks", {})
    registry.setdefault("steps", [])
    registry.setdefault("defaults", {})
    return registry


def save_registry(registry: dict, path_or_root) -> Path:
    ensure_registry_dirs(path_or_root)
    path = resolve_registry_path(path_or_root)
    payload = deepcopy(registry)
    payload["schema_version"] = SCHEMA_VERSION
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def init_registry(path_or_root, sequence_name: str | None = None, defaults: dict | None = None) -> tuple[dict, Path]:
    root = resolve_registry_root(path_or_root)
    ensure_registry_dirs(root)
    path = resolve_registry_path(root)
    if path.exists():
        registry = load_registry(path)
        if defaults:
            registry.setdefault("defaults", {})
            for key, value in defaults.items():
                if value not in (None, "") and registry["defaults"].get(key) in (None, ""):
                    registry["defaults"][key] = value
            save_registry(registry, root)
        return registry, path

    if sequence_name is None:
        sequence_name = root.name
    registry = default_registry(sequence_name=sequence_name, defaults=defaults)
    save_registry(registry, root)
    return registry, path


def next_step_idx(registry: dict) -> int:
    return len(registry.get("steps", [])) + 1


def find_latest_base_ckpt(runs_root: Path, view_id: int) -> Path | None:
    patterns = [
        runs_root / f"Cam{view_id}" / f"*cam{view_id}_*" / f"best_cam{view_id}.pth",
        runs_root / f"*cam{view_id}_*" / f"best_cam{view_id}.pth",
    ]
    candidates = []
    seen = set()
    for pattern in patterns:
        for path in runs_root.glob(str(pattern.relative_to(runs_root))):
            resolved = path.resolve()
            if resolved not in seen:
                candidates.append(path)
                seen.add(resolved)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.parent.stat().st_mtime, p.parent.name), reverse=True)
    return candidates[0]


def resolve_base_ckpt(
    task_id,
    *,
    view_id: int | None = None,
    base_ckpt: str | None = None,
    runs_root: str | Path = "outputs/rail_all",
) -> Path:
    if base_ckpt:
        path = resolve_artifact_path(base_ckpt)
        if not path.exists():
            raise FileNotFoundError(f"Base checkpoint not found: {path}")
        return path

    resolved_view_id = task_id_to_view_id(task_id, view_id=view_id)
    runs_root = resolve_artifact_path(runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"Baseline runs_root not found: {runs_root}")
    found = find_latest_base_ckpt(runs_root, resolved_view_id)
    if found is None:
        raise FileNotFoundError(
            f"Checkpoint not found for task_id={normalize_task_id(task_id)} view_id={resolved_view_id} under {runs_root}"
        )
    return found


def ensure_task_absent(registry: dict, task_id) -> str:
    normalized = normalize_task_id(task_id)
    if normalized in registry.get("tasks", {}):
        raise ValueError(f"task_id already exists in CAD registry: {normalized}")
    return normalized


def list_seen_task_ids(registry: dict, step_idx: int | None = None) -> list[str]:
    task_order = [normalize_task_id(task_id) for task_id in registry.get("task_order", [])]
    if step_idx is None:
        return task_order
    return task_order[: int(step_idx)]


def build_active_depth_peft_map(registry: dict) -> dict[int, str]:
    depth_map = {}
    for task_id in registry.get("task_order", []):
        task = registry.get("tasks", {}).get(str(task_id), {})
        peft_ckpt = task.get("active_peft_ckpt", "")
        view_id = task.get("view_id")
        if peft_ckpt and view_id is not None:
            depth_map[int(view_id)] = str(peft_ckpt)
    return depth_map


def export_active_depth_peft_map(registry: dict, path_or_root) -> Path:
    ensure_registry_dirs(path_or_root)
    root = resolve_registry_root(path_or_root)
    out_path = root / ACTIVE_MAP_FILENAME
    depth_map = {str(k): v for k, v in build_active_depth_peft_map(registry).items()}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(depth_map, f, ensure_ascii=False, indent=2)
    return out_path


def resolve_active_peft_ckpt(registry_or_path, *, task_id=None, view_id: int | None = None) -> str | None:
    registry = registry_or_path if isinstance(registry_or_path, dict) else load_registry(registry_or_path)
    normalized = normalize_task_id(task_id if task_id is not None else view_id)
    task = registry.get("tasks", {}).get(normalized)
    if not task:
        return None
    peft_ckpt = task.get("active_peft_ckpt", "")
    return str(peft_ckpt) if peft_ckpt else None


def register_task_step(
    registry: dict,
    *,
    task_id,
    view_id: int,
    base_ckpt,
    run_dir,
    peft_ckpt,
    backtest_dir,
    stage: str = "p1",
    created_at: str | None = None,
    baseline_eval: dict | None = None,
) -> dict:
    normalized = ensure_task_absent(registry, task_id)
    created_at = created_at or _now_text()
    step_idx = next_step_idx(registry)

    step_record = {
        "step_idx": int(step_idx),
        "task_id": normalized,
        "view_id": int(view_id),
        "stage": str(stage),
        "run_dir": to_registry_path_text(run_dir),
        "peft_ckpt": to_registry_path_text(peft_ckpt),
        "backtest_dir": to_registry_path_text(backtest_dir),
        "created_at": created_at,
    }

    task_record = {
        "task_id": normalized,
        "view_id": int(view_id),
        "base_ckpt": to_registry_path_text(base_ckpt),
        "active_peft_ckpt": to_registry_path_text(peft_ckpt),
        "active_run_dir": to_registry_path_text(run_dir),
        "baseline_eval": dict(baseline_eval or {}),
        "runs": [
            {
                "step_idx": int(step_idx),
                "stage": str(stage),
                "run_dir": to_registry_path_text(run_dir),
                "peft_ckpt": to_registry_path_text(peft_ckpt),
                "backtest_dir": to_registry_path_text(backtest_dir),
                "created_at": created_at,
            }
        ],
    }

    registry.setdefault("task_order", []).append(normalized)
    registry.setdefault("tasks", {})[normalized] = task_record
    registry.setdefault("steps", []).append(step_record)
    return step_record


def update_task_baseline_eval(registry: dict, task_id, baseline_eval: dict | None) -> None:
    normalized = normalize_task_id(task_id)
    task = registry.get("tasks", {}).get(normalized)
    if task is None:
        raise KeyError(f"task_id not found in registry: {normalized}")
    task["baseline_eval"] = dict(baseline_eval or {})


def snapshot_registry(registry: dict, path_or_root, step_idx: int | None = None) -> Path:
    ensure_registry_dirs(path_or_root)
    if step_idx is None:
        step_idx = next_step_idx(registry) - 1
    snapshot_path = registry_snapshot_dir(path_or_root) / f"registry_step{int(step_idx):03d}.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    return snapshot_path
