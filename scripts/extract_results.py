"""Materialize the gitignored result archives into `_extracted/`.

This is a convenience step so `scripts/generate_figures.py` and
`scripts/generate_reports.py` can be run from a clean clone without
having to manually unzip everything.
"""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "_extracted"


def _unzip(zip_path: Path) -> None:
    if not zip_path.exists():
        return
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(OUT)


def _unrar(rar_path: Path) -> None:
    """Best-effort RAR extraction. Falls back gracefully if no extractor is on PATH."""
    if not rar_path.exists():
        return
    candidates = [
        "unrar", "rar",
        r"C:\Program Files\WinRAR\UnRAR.exe",
        r"C:\Program Files\WinRAR\Rar.exe",
        r"C:\Program Files\7-Zip\7z.exe",
    ]
    for cand in candidates:
        try:
            cmd = [cand, "x", "-y", str(rar_path), str(OUT) + "/"]
            if cand.endswith("7z.exe"):
                cmd = [cand, "x", "-y", f"-o{OUT}", str(rar_path)]
            subprocess.run(cmd, check=True, capture_output=True)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    print(f"WARN: no RAR extractor found; skipped {rar_path.name}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for z in ["metrics.zip", "tables.zip", "logs.zip",
              "figures.zip", "batch_predictions.zip"]:
        _unzip(RESULTS / z)

    # nested rar files
    for r in ["figures.rar", "batch_predictions.rar"]:
        _unrar(OUT / r)
        # remove the intermediate .rar so the directory is clean
        rp = OUT / r
        if rp.exists():
            rp.unlink()

    # checkpoints.zip is huge (~793 MB) — skip by default
    print(f"Extracted under: {OUT}")


if __name__ == "__main__":
    main()
