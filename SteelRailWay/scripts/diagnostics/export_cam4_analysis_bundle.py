"""Export a compact Cam4 analysis bundle into the repo-level results directory."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "rail_ablation"
RESULTS_DIR = ROOT / "results"

PEFT_SCOPE_SUMMARY = OUTPUT_ROOT / "cam4_peft_scope_audit" / "summary.txt"
PEFT_CFCA_ROOT = OUTPUT_ROOT / "cam4_cf_ca_peft"
PEFT_CFCA_SCHEMES = ["full", "no_ca", "no_cf", "no_cf_ca"]
PEFT_CFCA_SCORE_SOURCES = ["rgb", "depth", "fusion", "rgb_isolated", "depth_isolated"]
PEFT_CFCA_REPAIR_ROOT = OUTPUT_ROOT / "cam4_cfca_repair"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_scope_summary() -> Path:
    out_path = RESULTS_DIR / "cam4_peft_scope_audit_summary.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PEFT_SCOPE_SUMMARY, out_path)
    return out_path


def merge_peft_cfca_scores() -> list[Path]:
    written_paths: list[Path] = []
    master_rows: list[dict] = []

    for scheme in PEFT_CFCA_SCHEMES:
        scheme_dir = PEFT_CFCA_ROOT / scheme
        result = read_json(scheme_dir / "result.json")

        merged: dict[tuple[str, str], dict] = {}
        for source in PEFT_CFCA_SCORE_SOURCES:
            rows = read_csv_rows(scheme_dir / f"scores_{source}.csv")
            for row in rows:
                key = (row["frame_id"], row["label"])
                item = merged.setdefault(
                    key,
                    {
                        "module_ablation": scheme,
                        "frame_id": row["frame_id"],
                        "label": row["label"],
                        "fusion_auroc_total": f"{float(result['auroc']):.8f}",
                        "rgb_auroc": f"{float(result['auroc_by_source']['rgb']):.8f}",
                        "depth_auroc": f"{float(result['auroc_by_source']['depth']):.8f}",
                        "fusion_auroc": f"{float(result['auroc_by_source']['fusion']):.8f}",
                        "rgb_isolated_auroc": f"{float(result['auroc_by_source']['rgb_isolated']):.8f}",
                        "depth_isolated_auroc": f"{float(result['auroc_by_source']['depth_isolated']):.8f}",
                    },
                )
                item[f"{source}_rank"] = row["rank"]
                item[f"{source}_score"] = row["score"]

        merged_rows = sorted(merged.values(), key=lambda x: (x["label"], x["frame_id"]))
        fieldnames = [
            "module_ablation",
            "frame_id",
            "label",
            "fusion_auroc_total",
            "rgb_auroc",
            "depth_auroc",
            "fusion_auroc",
            "rgb_isolated_auroc",
            "depth_isolated_auroc",
            "rgb_rank",
            "rgb_score",
            "depth_rank",
            "depth_score",
            "fusion_rank",
            "fusion_score",
            "rgb_isolated_rank",
            "rgb_isolated_score",
            "depth_isolated_rank",
            "depth_isolated_score",
        ]
        out_path = RESULTS_DIR / f"cam4_cf_ca_peft_scores_{scheme}_merged.csv"
        write_csv(out_path, merged_rows, fieldnames)
        written_paths.append(out_path)
        master_rows.extend(merged_rows)

    master_path = RESULTS_DIR / "cam4_cf_ca_peft_scores_all_merged.csv"
    write_csv(
        master_path,
        master_rows,
        [
            "module_ablation",
            "frame_id",
            "label",
            "fusion_auroc_total",
            "rgb_auroc",
            "depth_auroc",
            "fusion_auroc",
            "rgb_isolated_auroc",
            "depth_isolated_auroc",
            "rgb_rank",
            "rgb_score",
            "depth_rank",
            "depth_score",
            "fusion_rank",
            "fusion_score",
            "rgb_isolated_rank",
            "rgb_isolated_score",
            "depth_isolated_rank",
            "depth_isolated_score",
        ],
    )
    written_paths.append(master_path)

    summary_rows = []
    for scheme in PEFT_CFCA_SCHEMES:
        result = read_json(PEFT_CFCA_ROOT / scheme / "result.json")
        summary_rows.append(
            {
                "module_ablation": scheme,
                "fusion_auroc_total": f"{float(result['auroc']):.8f}",
                "rgb_auroc": f"{float(result['auroc_by_source']['rgb']):.8f}",
                "depth_auroc": f"{float(result['auroc_by_source']['depth']):.8f}",
                "fusion_auroc": f"{float(result['auroc_by_source']['fusion']):.8f}",
                "rgb_isolated_auroc": f"{float(result['auroc_by_source']['rgb_isolated']):.8f}",
                "depth_isolated_auroc": f"{float(result['auroc_by_source']['depth_isolated']):.8f}",
                "result_json": str((PEFT_CFCA_ROOT / scheme / "result.json").relative_to(ROOT)),
            }
        )
    summary_path = RESULTS_DIR / "cam4_cf_ca_peft_scores_auroc_summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        [
            "module_ablation",
            "fusion_auroc_total",
            "rgb_auroc",
            "depth_auroc",
            "fusion_auroc",
            "rgb_isolated_auroc",
            "depth_isolated_auroc",
            "result_json",
        ],
    )
    written_paths.append(summary_path)
    return written_paths


def merge_cfca_repair_results() -> list[Path]:
    top_summary_rows = read_csv_rows(PEFT_CFCA_REPAIR_ROOT / "summary.csv")
    merged_rows: list[dict] = []

    for row in top_summary_rows:
        final_ckpt = Path(row["final_ckpt"])
        run_dir = ROOT / final_ckpt.parent.parent
        fold_summary_path = run_dir / "summary.csv"
        final_result_path = run_dir / "final" / "eval" / "result.json"
        final_result = read_json(final_result_path)

        merged_rows.append(
            {
                "scope": row["scheme"],
                "row_type": "final",
                "fold": "final",
                "rgb_auroc": row["rgb_auroc"],
                "depth_auroc": row["depth_auroc"],
                "fusion_auroc": row["fusion_auroc"],
                "delta_vs_peft_full": row["delta_vs_peft_full"],
                "trainable_rgb_count": row["trainable_rgb_count"],
                "trainable_depth_count": row["trainable_depth_count"],
                "best_epoch": str(final_result.get("best_epoch", "")),
                "best_val_loss": str(final_result.get("best_val_loss", "")),
                "train_good": "",
                "test_good": "",
                "ckpt": row["final_ckpt"],
                "source_run_dir": str(run_dir.relative_to(ROOT)),
            }
        )

        for fold_row in read_csv_rows(fold_summary_path):
            merged_rows.append(
                {
                    "scope": fold_row["scope"],
                    "row_type": "cv_fold",
                    "fold": fold_row["fold"],
                    "rgb_auroc": fold_row["rgb_auroc"],
                    "depth_auroc": fold_row["depth_auroc"],
                    "fusion_auroc": fold_row["fusion_auroc"],
                    "delta_vs_peft_full": "",
                    "trainable_rgb_count": row["trainable_rgb_count"],
                    "trainable_depth_count": row["trainable_depth_count"],
                    "best_epoch": "",
                    "best_val_loss": "",
                    "train_good": fold_row["train_good"],
                    "test_good": fold_row["test_good"],
                    "ckpt": fold_row["ckpt"],
                    "source_run_dir": str(run_dir.relative_to(ROOT)),
                }
            )

    merged_path = RESULTS_DIR / "cam4_cfca_repair_merged.csv"
    write_csv(
        merged_path,
        merged_rows,
        [
            "scope",
            "row_type",
            "fold",
            "rgb_auroc",
            "depth_auroc",
            "fusion_auroc",
            "delta_vs_peft_full",
            "trainable_rgb_count",
            "trainable_depth_count",
            "best_epoch",
            "best_val_loss",
            "train_good",
            "test_good",
            "ckpt",
            "source_run_dir",
        ],
    )

    return [merged_path]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written.append(copy_scope_summary())
    written.extend(merge_peft_cfca_scores())
    written.extend(merge_cfca_repair_results())

    print("Exported files:")
    for path in written:
        print(f"- {path}")


if __name__ == "__main__":
    main()
