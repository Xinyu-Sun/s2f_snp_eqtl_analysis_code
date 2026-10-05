#!/usr/bin/env python3
"""Render manuscript Figure S1 from the model-convergence summary table.

Input (in $S2F_RESULTS): model_convergence_results.tsv. Output (in $S2F_FIGURE_OUTPUT):
model_convergence_heatmaps_manuscript.pdf/.png. Plotting code: model_convergence_plotting.py."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import pandas as pd


RESULTS = Path(os.environ["S2F_RESULTS"])
OUT = Path(os.environ.get("S2F_FIGURE_OUTPUT", str(RESULTS / "figures")))
SOURCE = Path(__file__).resolve().parent / "model_convergence_plotting.py"


def load_plotter():
    spec = importlib.util.spec_from_file_location("s2f_model_convergence", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plotting module: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plot_convergence_heatmaps


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(RESULTS / "model_convergence_results.tsv", sep="\t")
    load_plotter()(results, OUT)

    source_pdf = OUT / "model_convergence_heatmaps.pdf"
    source_png = OUT / "model_convergence_heatmaps.png"
    target_pdf = OUT / "model_convergence_heatmaps_manuscript.pdf"
    target_png = OUT / "model_convergence_heatmaps_manuscript.png"
    shutil.copy2(source_pdf, target_pdf)
    shutil.copy2(source_png, target_png)
    print(f"[OK] Saved {target_pdf}")


if __name__ == "__main__":
    main()
