"""
run_simulation.py

Main entry point. Runs the full CTSF empirical simulation described in
paper Section 5.5:

  1. Train baseline anomaly detectors (one per signal dimension)
  2. Reconstruct the four documented incidents and compute detection points
  3. Evaluate classifier performance on a held-out synthetic test set
  4. Generate the trust-score trajectory figure

Usage:
    python src/run_simulation.py
"""

import os
import numpy as np
from ctsf_model import CTSFEngine, gen_baseline_sessions
from case_reconstructions import run_all_cases
from classifier_validation import build_test_set, evaluate
from plot_trajectories import plot_all


def main():
    # Single global seed, matching the order of random draws used to
    # produce the figures reported in the paper (Section 5.5). Do not
    # reseed partway through -- baseline generation, case reconstruction,
    # and test-set generation all draw from this one continuous stream.
    np.random.seed(42)

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("Step 1: Generating baseline sessions and training signal detectors...")
    baseline = gen_baseline_sessions("payroll_analyst", n=800)
    baseline.to_csv("data/baseline_sessions.csv", index=False)
    engine = CTSFEngine(baseline)

    print("\nStep 2: Reconstructing documented incidents...")
    summary = run_all_cases(engine)
    print(summary.to_string(index=False))

    print("\nStep 3: Evaluating classifier on held-out synthetic test set...")
    test = build_test_set(n_normal=400, n_attack=100)
    metrics = evaluate(engine, test)
    print(f"Precision: {metrics['precision']:.3f}  "
          f"Recall: {metrics['recall']:.3f}  "
          f"F1: {metrics['f1']:.3f}  "
          f"FPR: {metrics['fpr']:.3f}  "
          f"(n={metrics['n']})")

    print("\nStep 4: Generating trust score trajectory figure...")
    plot_all()

    print("\nDone. See results/ for CSVs and figures/ for the trajectory plot.")


if __name__ == "__main__":
    main()
