"""Orchestrate the entire GETE reproduction pipeline.

Usage
-----
python run_all.py --stages all
python run_all.py --stages transformer,gnn,ensemble,evaluate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STAGES = {
    "transformer": [sys.executable, "-m", "train.train_transformer"],
    "gnn":         [sys.executable, "-m", "train.train_gnn"],
    "ensemble":    [sys.executable, "-m", "train.train_ensemble"],
    "evaluate":    [sys.executable, "evaluate.py"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", type=str, default="all",
                   help="comma-separated subset of " + ",".join(STAGES) + " or 'all'")
    p.add_argument("--extra", nargs=argparse.REMAINDER,
                   help="extra args passed to every sub-command")
    args = p.parse_args()

    names = list(STAGES) if args.stages == "all" else args.stages.split(",")
    for name in names:
        if name not in STAGES:
            raise SystemExit(f"unknown stage: {name}")
        cmd = STAGES[name] + (args.extra or [])
        print(f"\n===== stage: {name} =====")
        print(" ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise SystemExit(f"stage {name} failed (rc={rc})")


if __name__ == "__main__":
    main()
