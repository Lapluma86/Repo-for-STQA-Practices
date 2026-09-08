"""Continual-learning metric helpers for CAD backtests."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _score_from_eval_block(block: dict, metric_source: str = "fusion"):
    if not block:
        return None
    metric_map = block.get("auroc_by_source") or {}
    value = metric_map.get(metric_source)
    if value is None and metric_source == "fusion":
        value = block.get("auroc")
    return None if value is None else float(value)


def build_matrix_payload(step_payloads: list[dict], task_order: list[str], metric_source: str = "fusion") -> dict:
    task_order = [str(task_id) for task_id in task_order]
    steps = []
    for payload in sorted(step_payloads, key=lambda item: int(item.get("step_idx", 0))):
        row_map = {task_id: None for task_id in task_order}
        for row in payload.get("rows", []):
            task_id = str(row.get("task_id"))
            if task_id not in row_map:
                continue
            row_map[task_id] = _score_from_eval_block(row.get("cad_active", {}), metric_source=metric_source)
        steps.append(
            {
                "step_idx": int(payload.get("step_idx", 0)),
                "task_id": str(payload.get("current_task_id", "")),
                "scores": row_map,
            }
        )
    return {
        "metric_source": str(metric_source),
        "task_order": task_order,
        "steps": steps,
    }


def compute_continual_metrics(matrix_payload: dict) -> dict:
    task_order = list(matrix_payload.get("task_order", []))
    steps = list(matrix_payload.get("steps", []))
    intro_step = {task_id: idx + 1 for idx, task_id in enumerate(task_order)}
    matrix = {int(step["step_idx"]): dict(step.get("scores", {})) for step in steps}

    per_step = []
    for step in steps:
        step_idx = int(step["step_idx"])
        seen = task_order[:step_idx]
        current_scores = matrix.get(step_idx, {})
        evaluable = [current_scores.get(task_id) for task_id in seen if current_scores.get(task_id) is not None]
        acc = None if not evaluable else float(sum(evaluable) / len(evaluable))

        historical = seen[:-1]
        bwt_terms = []
        forgetting_terms = []
        retention_terms = []
        for task_id in historical:
            current = current_scores.get(task_id)
            anchor = matrix.get(intro_step[task_id], {}).get(task_id)
            if current is not None and anchor is not None:
                bwt_terms.append(float(current - anchor))

            observed = []
            for back_step in range(intro_step[task_id], step_idx + 1):
                value = matrix.get(back_step, {}).get(task_id)
                if value is not None:
                    observed.append(float(value))
            if current is None or not observed:
                continue
            best = max(observed)
            forgetting = max(0.0, best - float(current))
            forgetting_terms.append(float(forgetting))
            if best > 0:
                retention_terms.append(float(current) / float(best))

        per_step.append(
            {
                "step_idx": step_idx,
                "task_id": str(step.get("task_id", "")),
                "seen_task_ids": list(seen),
                "acc": acc,
                "bwt": 0.0 if not bwt_terms else float(sum(bwt_terms) / len(bwt_terms)),
                "avg_forgetting": 0.0 if not forgetting_terms else float(sum(forgetting_terms) / len(forgetting_terms)),
                "max_forgetting": 0.0 if not forgetting_terms else float(max(forgetting_terms)),
                "retention": 1.0 if not retention_terms else float(sum(retention_terms) / len(retention_terms)),
                "fwt": 0.0,
            }
        )

    current_step = per_step[-1] if per_step else None
    return {
        "metric_source": matrix_payload.get("metric_source", "fusion"),
        "task_order": task_order,
        "per_step": per_step,
        "current_step": current_step,
    }


def write_matrix_csv(path: Path, matrix_payload: dict) -> None:
    task_order = list(matrix_payload.get("task_order", []))
    fields = ["step_idx", "task_id"] + [f"task_{task_id}" for task_id in task_order]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in matrix_payload.get("steps", []):
            row = {
                "step_idx": int(step.get("step_idx", 0)),
                "task_id": str(step.get("task_id", "")),
            }
            for task_id in task_order:
                value = step.get("scores", {}).get(task_id)
                row[f"task_{task_id}"] = "" if value is None else f"{float(value):.8f}"
            writer.writerow(row)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
