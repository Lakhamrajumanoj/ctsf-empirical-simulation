# CTSF Empirical Simulation

Simulation code supporting Section 5.5 ("Empirical Simulation of CTSF Detection
Performance") of:

> Lakhamraju, M. V. *Securing the Digital Core: A Cognitive Trust Scoring
> Framework for ERP System Security.*

This repository implements the Cognitive Trust Scoring Framework (CTSF)
Layer 3 behavioral trust engine and evaluates it against reconstructed
timelines of four publicly documented ERP-related security incidents
(Twilio 2022, MGM Resorts 2023, BenefitMall 2018, "Payroll Pirates"
2023–2025), plus a held-out synthetic classifier validation.

## Important scope note

**This is a simulation calibrated to publicly reported facts about each
incident, not raw telemetry from the breached organizations.** No real
organizational data, session logs, or user data of any kind is used
anywhere in this repository. Baseline "normal" session behavior is
synthetically generated to reflect plausible role-based usage patterns.
See the paper's Section 8 (Limitations) for the caveats this implies.

## Repository structure

```
.
├── src/
│   ├── ctsf_model.py            # Core engine: signal detectors, weighted TS(t) formula
│   ├── case_reconstructions.py  # Reconstructs the four documented incidents
│   ├── classifier_validation.py # Precision/recall/F1/FPR on held-out test set
│   ├── plot_trajectories.py     # Generates the trust-score trajectory figure
│   └── run_simulation.py        # Main entry point — runs everything end to end
├── data/                        # Generated baseline session data (created on run)
├── results/                     # Generated CSV outputs (created on run)
├── figures/                     # Generated figure (created on run)
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the simulation

```bash
python src/run_simulation.py
```

This will:
1. Generate 800 synthetic baseline ("normal") sessions and train one
   Isolation Forest detector per signal dimension (temporal, behavioural,
   volumetric, geo-device, role-norm).
2. Reconstruct each of the four documented incidents as a time series of
   sessions and compute the trust score TS(t) at each step, using the
   exact weights and thresholds defined in the paper (Section 4.4).
3. Evaluate the resulting classifier on a held-out synthetic test set
   (400 normal + 100 attack-pattern sessions).
4. Save all trajectories and summary tables to `results/`, and the
   trajectory figure to `figures/`.

Expected console output (with the fixed random seed used throughout):

```
          case    unit  total_steps_simulated  ctsf_detection_point_simulated  TS_at_start  TS_at_final_step
   BenefitMall     day                    130                               8        0.942             0.011
           MGM    hour                     72                               5        0.873             0.147
        Twilio session                      6                               1        0.772             0.186
PayrollPirates session                      3                               1        0.772             0.186

Precision: 0.971  Recall: 1.000  F1: 0.985  FPR: 0.007  (n=500)
```

These numbers correspond directly to Table 5 and the classifier
validation figures reported in paper Section 5.5.

## Reproducibility

All randomness is seeded once (`np.random.seed(42)`) at the top of
`run_simulation.py`. Baseline generation, case reconstruction, and
test-set generation all draw from this single continuous random stream
in a fixed order — do not reorder the calls in `main()` or reseed
partway through, or the reported numbers will no longer match the paper.

## Model summary

| Component | Value |
|---|---|
| Signal weights | w₁=0.15 (temporal), w₂=0.20 (behavioural), w₃=0.25 (volumetric), w₄=0.25 (geo-device), w₅=0.15 (role-norm) |
| Decay function | α(t) = e^(−λ·Δt) |
| Trust score | TS(t) = 1 − α(t)·A(t) |
| Full access | TS ≥ 0.80 |
| Re-authentication | 0.60 ≤ TS < 0.80 |
| Read-only demotion | 0.40 ≤ TS < 0.60 |
| Terminate + alert | TS < 0.40 |

## Extending this work

Documented next steps (see paper Section 8):
- Replace synthetic baseline sessions with real (anonymized) organizational
  session logs.
- Empirically fit the decay constant λ rather than assuming a fixed value.
- Validate against a live or sandboxed ERP deployment rather than a
  reconstructed timeline.

## License

MIT — see `LICENSE`.
