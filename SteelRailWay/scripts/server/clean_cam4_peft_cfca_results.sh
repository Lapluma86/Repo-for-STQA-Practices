#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

rm -rf outputs/rail_ablation/cam4_cf_ca_peft
rm -rf outputs/rail_ablation/cam4_cf_ca_peft_fusion_rules
rm -rf outputs/rail_ablation/cam4_peft_scope_audit
rm -rf outputs/rail_ablation/cam4_cfca_repair
rm -rf outputs/rail_ablation/cam4_cfca_repair_smoke
rm -rf outputs/rail_ablation/cam4_case_groups/analysis

echo "[DONE] Cleaned PEFT CF/CA result directories."
