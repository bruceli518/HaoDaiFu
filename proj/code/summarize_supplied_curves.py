"""Compute ALC summaries from the four learning-curve CSVs supplied with the project."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ["BOW", "CBOW", "CBOW_KG", "BOW_EXP", "BOW_EXP_KG", "LSTM", "LSTM_KG", "BERT"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves-dir", default="../../results")
    args = parser.parse_args()
    root = Path(args.curves_dir)
    summaries = []
    for dataset, subset in [("haodf", "all"), ("haodf", "rare"), ("chinare", "all"), ("chinare", "rare")]:
        path = root / dataset / "learning_curves" / f"{subset}.csv"
        curve = pd.read_csv(path, header=None, encoding="utf-8-sig", names=METHODS)
        rates = np.arange(0.1, 1.01, 0.1)
        for method in METHODS:
            summaries.append({"dataset": dataset, "subset": subset, "method": method,
                              "alc": np.trapezoid(curve[method], rates), "f1_at_10pct": curve[method].iloc[0],
                              "f1_at_100pct": curve[method].iloc[-1]})
    result = pd.DataFrame(summaries)
    result.to_csv(root / "supplied_curve_alc.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
