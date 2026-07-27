"""
sensitivity_analysis.py

Weight sensitivity and ablation analysis for the CTSF trust-scoring
model, supporting the paper's response to peer review (Reviewer B:
"Have you conducted sensitivity analysis or ablation studies?").

This script reuses the same CTSFEngine, baseline generation, and
held-out test set construction as run_simulation.py, so results are
directly comparable to the main classifier validation numbers reported
in Section 6.

It evaluates:
  1. The original paper weights (baseline for comparison)
  2. Equal weighting across all five signals (naive baseline)
  3. Leave-one-signal-out ablation (each signal zeroed, its weight
     redistributed equally among the remaining four)
  4. +/-50% perturbation of each individual weight, renormalized so
     all weights still sum to 1

Usage:
    python src/sensitivity_analysis.py

Must be run from the repository root (so it can import from src/ and
write to results/), same as run_simulation.py.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from ctsf_model import CTSFEngine, gen_baseline_sessions, WEIGHTS, FEATURE_MAP
from classifier_validation import build_test_set


def score_all_signals(engine: CTSFEngine, df: pd.DataFrame) -> dict:
    """
    Vectorized computation of all five signal sub-scores for an entire
    DataFrame at once (much faster than scoring row by row, and the
    sub-scores don't depend on the weights, so this only needs to run
    once regardless of how many weight configurations are tested).
    """
    scores = {}
    for sig, col in FEATURE_MAP.items():
        X = df[[col]].values
        clf = engine.forests[sig]
        raw = -clf.decision_function(X)
        lo, hi = engine.score_ranges[sig]
        scores[sig] = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)
    return scores


def evaluate_weight_config(test: pd.DataFrame, subscores: dict,
                            weights: dict, flag_threshold: float = 0.60) -> dict:
    """Compute classifier metrics for a given weight configuration."""
    A = sum(weights[sig] * subscores[sig] for sig in weights)
    TS = 1 - A
    pred = (TS < flag_threshold).astype(int)
    label = test["label"].values

    return {
        "precision": round(precision_score(label, pred), 3),
        "recall": round(recall_score(label, pred), 3),
        "f1": round(f1_score(label, pred), 3),
        "fpr": round(((pred == 1) & (label == 0)).sum() / (label == 0).sum(), 3),
    }


def run_ablation(engine: CTSFEngine, test: pd.DataFrame) -> pd.DataFrame:
    subscores = score_all_signals(engine, test)
    results = []

    # 1. Original paper weights
    metrics = evaluate_weight_config(test, subscores, WEIGHTS)
    results.append({"configuration": "Original weights (paper)", **metrics})

    # 2. Equal weighting (naive baseline)
    equal_weights = {k: 0.20 for k in WEIGHTS}
    metrics = evaluate_weight_config(test, subscores, equal_weights)
    results.append({"configuration": "Equal weights (0.20 each)", **metrics})

    # 3. Leave-one-signal-out ablation
    for dropped_signal in WEIGHTS:
        w = dict(WEIGHTS)
        removed_weight = w.pop(dropped_signal)
        for k in w:
            w[k] += removed_weight / len(w)
        w[dropped_signal] = 0.0
        metrics = evaluate_weight_config(test, subscores, w)
        results.append({"configuration": f"Drop {dropped_signal} signal", **metrics})

    # 4. +/-50% perturbation of each individual weight, renormalized
    for signal in WEIGHTS:
        for multiplier, tag in [(1.5, "+50%"), (0.5, "-50%")]:
            w = dict(WEIGHTS)
            w[signal] = w[signal] * multiplier
            total = sum(w.values())
            w = {k: v / total for k, v in w.items()}
            metrics = evaluate_weight_config(test, subscores, w)
            results.append({"configuration": f"{signal} {tag} (renormalized)", **metrics})

    return pd.DataFrame(results)


def main():
    import os
    os.makedirs("results", exist_ok=True)

    # Same seeding order as run_simulation.py, so this can be run
    # standalone and still reproduce the paper's reported numbers.
    np.random.seed(42)

    baseline = gen_baseline_sessions("payroll_analyst", n=800)
    engine = CTSFEngine(baseline)
    test = build_test_set(n_normal=400, n_attack=100)

    print("Running weight sensitivity and ablation analysis...\n")
    results = run_ablation(engine, test)
    print(results.to_string(index=False))

    results.to_csv("results/sensitivity_ablation.csv", index=False)
    print("\nSaved to results/sensitivity_ablation.csv")


if __name__ == "__main__":
    main()
