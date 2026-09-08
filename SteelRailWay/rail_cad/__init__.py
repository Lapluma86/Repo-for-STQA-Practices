"""Continual-adaptation helpers for task-aware per-view DepthAffinePEFT."""

from .metrics import build_matrix_payload, compute_continual_metrics, write_matrix_csv
from .registry import (
    build_active_depth_peft_map,
    export_active_depth_peft_map,
    find_latest_base_ckpt,
    init_registry,
    list_seen_task_ids,
    load_registry,
    register_task_step,
    resolve_active_peft_ckpt,
    resolve_artifact_path,
    resolve_base_ckpt,
    resolve_registry_path,
    save_registry,
    snapshot_registry,
    task_id_to_view_id,
    update_task_baseline_eval,
)


def add_p1_training_args(*args, **kwargs):
    from .p1 import add_p1_training_args as _add_p1_training_args

    return _add_p1_training_args(*args, **kwargs)


def run_p1_experiment(*args, **kwargs):
    from .p1 import run_p1_experiment as _run_p1_experiment

    return _run_p1_experiment(*args, **kwargs)

__all__ = [
    "add_p1_training_args",
    "build_active_depth_peft_map",
    "build_matrix_payload",
    "compute_continual_metrics",
    "export_active_depth_peft_map",
    "find_latest_base_ckpt",
    "init_registry",
    "list_seen_task_ids",
    "load_registry",
    "register_task_step",
    "resolve_active_peft_ckpt",
    "resolve_artifact_path",
    "resolve_base_ckpt",
    "resolve_registry_path",
    "run_p1_experiment",
    "save_registry",
    "snapshot_registry",
    "task_id_to_view_id",
    "update_task_baseline_eval",
    "write_matrix_csv",
]
