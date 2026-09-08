#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compatibility wrapper for the original Cam4 P1 experiment."""

# >>> path-bootstrap >>>
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
# <<< path-bootstrap <<<

import argparse

from rail_cad.p1 import add_p1_training_args, run_p1_experiment


def build_parser():
    parser = argparse.ArgumentParser(
        description="Cam4 P1: train a 2-parameter depth affine PEFT with 4-fold CV."
    )
    add_p1_training_args(
        parser,
        ckpt_required=True,
        default_output_root="outputs/rail_peft",
        default_view_id=4,
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.task_id is None:
        args.task_id = str(int(args.view_id))
    run_p1_experiment(
        args,
        summary_title=f"Cam{args.view_id} DepthAffinePEFT P1",
        emit_depth_peft_map=True,
    )


if __name__ == "__main__":
    main()
