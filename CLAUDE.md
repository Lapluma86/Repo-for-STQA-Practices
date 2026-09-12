# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **Software Testing and Quality Assurance (STQA) Practices** repository containing multiple deep learning projects under active development and testing.

### Projects

| Project | Description |
|---------|-------------|
| `SteelRailWay/` | Rail defect detection using multimodal (RGB + Depth) Teacher Reverse Distillation (TRD) |
| `AdoDAS/` | Depression/Anxiety/Stress prediction from audio-visual data (has its own CLAUDE.md) |

---

## SteelRailWay Project

### Project Purpose

Rail defect detection system using multimodal TRD (Teacher Reverse Distillation) architecture. Processes ultra-long strip images (6000x900) via patch sliding window (7 patches of 900x900 each).

### Architecture

**TRD Framework**: Teacher Reverse Distillation uses a pretrained encoder as teacher, with a student decoder trained to match multi-scale feature representations.

```
Input Image → Patch Extraction → Encoder (Teacher) → Multi-scale Features
                                              ↓
                                         Decoder (Student)
                                              ↓
                                    Feature Matching Loss
```

**Key Components**:
- `datasets/rail_dataset.py`: Dual-modal (RGB + Depth) dataset with patch sliding
- `models/trd/`: Teacher-Student architecture (encoder, decoder, projector)
- `models/peft/`: Parameter-efficient fine-tuning adapters (Adapter, ViewPrompt, ViewFiLM)
- `utils/losses.py`:蒸馏损失函数 (L2, Cosine, Pixel-wise)
- `eval/`: Evaluation metrics (I-AUC, P-AUC, PRO, engineering metrics)

### Data Organization

```
train/val (data_20260327/):
    Cam{1-8}/
        rgb/*.jpg     (~903×6000)
        depth/*.tiff

test (rail_mvtec_gt_test/):
    rail_mvtec/cam{1-6}/
        test/good/*.jpg     normal samples
        test/broken/*.jpg   anomaly samples
        ground_truth/broken/*.png   pixel-level masks
    rail_mvtec_depth/cam{1-6}/
        test/good/*.tiff
        test/broken/*.tiff
```

### Running Training

```bash
cd SteelRailWay

# Rail dataset training
python train/train_trd_rail.py --train_root <path> --test_root <path> --view_id 1

# MVTec3D RGB-D training
python train/train_trd_mvtec3d_rgbd.py --data_root <path>

# Eyecandies dataset
python train/train_trd_eyecandies.py --data_root <path>
```

### Testing Framework

```bash
cd SteelRailWay

# Install test dependencies
pip install -r tests/requirements-test.txt

# Run all tests
pytest tests/manual/phase1/ -v

# Run blackbox tests only
pytest tests/manual/phase1/blackbox/ -v

# Run whitebox tests only
pytest tests/manual/phase1/whitebox/ -v

# Generate coverage report
pytest tests/manual/phase1/whitebox/ \
    --cov=datasets.rail_dataset \
    --cov=utils.losses \
    --cov=eval.metrics_engineering \
    --cov-report=html:tests/manual/phase1/coverage_reports \
    -v
```

### Test Structure

```
tests/
├── manual/phase1/
│   ├── blackbox/          # 85 test cases
│   │   ├── test_equivalence_partitioning.py
│   │   ├── test_boundary_value.py
│   │   ├── test_decision_table.py
│   │   └── test_state_transition.py
│   └── whitebox/          # 85 test cases
│       ├── test_statement_coverage.py
│       ├── test_branch_coverage.py
│       ├── test_condition_coverage.py
│       └── test_path_coverage.py
├── automated/             # Module 2: automation tests
├── performance/           # Module 2: performance tests
├── security/             # Module 2: security tests
└── integration/          # Module 2: integration tests
```

### Key Parameters Requiring Validation

The dataset class validates these parameters in `__init__`:
- `view_id`: 1-8 (test set only has cam1-6)
- `split`: "train" | "val" | "test"
- `img_size`: positive integer
- `depth_norm`: "zscore" | "minmax" | "log"
- `patch_size`: positive integer
- `patch_stride`: positive integer (>0 to prevent infinite loops)
- `train_sample_ratio`: 0-1 range

### Critical Files for Testing

| File | Purpose |
|------|---------|
| `datasets/rail_dataset.py` | Dataset with parameter validation |
| `utils/losses.py` | Loss functions (handles empty list edge case) |
| `eval/metrics_engineering.py` | Engineering metrics |
| `train/train_trd_rail.py` | Main training script |

---

## Development Workflow

### Before Modifying Core Files

1. Check existing tests in `tests/manual/phase1/`
2. Understand the parameter validation rules in `datasets/rail_dataset.py`
3. Review loss function edge cases in `utils/losses.py`

### After Bug Fixes

1. Update corresponding test cases to match new validation behavior
2. Run pytest to verify fixes don't break existing tests
3. Update `tests/BUG_FIX_REPORT.md` with fix details

### Code Style

- Use type hints for function parameters
- Add docstrings explaining parameters and return values
- Handle edge cases explicitly rather than relying on silent failures

---

## Working with This Repository

### Environment Notes

- This is a PyTorch-based deep learning project
- CUDA/GPU recommended for training
- Tests can run on CPU but may skip GPU-specific tests

### Common Issues

1. **Data path errors**: Verify train_root and test_root directories exist with correct structure
2. **CUDA OOM**: Reduce batch_size or use AMP (mixed precision) training
3. **Test failures**: Check if data directories are accessible

### Documentation

- `tests/PROJECT_OVERVIEW.md` - Test project overview
- `tests/WORK_COMPLETION_REPORT.md` - Phase 1 completion report
- `tests/BUG_FIX_REPORT.md` - Fixed defects documentation
- `tests/FUTURE_PLAN.md` - Testing roadmap
