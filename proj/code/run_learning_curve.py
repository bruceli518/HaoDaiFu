"""Run the paper's per-class learning curve protocol on an authorized corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models import build_model

RATES = np.arange(0.1, 1.01, 0.1)


def per_class_sample(frame: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    """Sample a non-empty, nested-size training set for every disease class."""
    sampled = [
        group.sample(n=max(1, int(np.floor(len(group) * rate))), random_state=seed)
        for _, group in frame.groupby("label")
    ]
    return pd.concat(sampled, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="UTF-8 CSV with text,label columns")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["bow", "bow_exp", "bert"])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    data = pd.read_csv(args.data, encoding="utf-8-sig").dropna(subset=["text", "label"])
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("Input CSV must include text,label columns")
    if (data.groupby("label").size() < 2).any():
        raise ValueError("Every label requires at least two records for a stratified train/test split")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for repeat in range(args.repeats):
        train, test = train_test_split(data, test_size=0.2, stratify=data.label, random_state=args.seed + repeat)
        for rate in tqdm(RATES, desc=f"split {repeat + 1}/{args.repeats}"):
            subset = per_class_sample(train, rate, args.seed + repeat)
            for model_name in args.models:
                model = build_model(model_name, seed=args.seed + repeat).fit(subset.text, subset.label)
                macro_f1 = f1_score(test.label, model.predict(test.text), average="macro", zero_division=0)
                rows.append({"repeat": repeat, "rate": rate, "model": model_name, "macro_f1": macro_f1,
                             "train_documents": len(subset), "mean_documents_per_class": len(subset) / subset.label.nunique()})

    raw = pd.DataFrame(rows)
    curve = raw.groupby(["model", "rate"], as_index=False).agg(macro_f1=("macro_f1", "mean"), std=("macro_f1", "std"), mean_documents_per_class=("mean_documents_per_class", "mean"))
    alc = curve.groupby("model").apply(lambda x: np.trapezoid(x.macro_f1, x.rate)).rename("alc").reset_index()
    raw.to_csv(output / "per_repeat_scores.csv", index=False)
    curve.to_csv(output / "learning_curve.csv", index=False)
    alc.to_csv(output / "alc.csv", index=False)
    (output / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    print(curve.to_string(index=False))
    print("\nALC\n" + alc.to_string(index=False))


if __name__ == "__main__":
    main()
