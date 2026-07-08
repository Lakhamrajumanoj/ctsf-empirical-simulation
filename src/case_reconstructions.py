"""
case_reconstructions.py

Reconstructs each documented incident discussed in the paper (Section 5)
as a time series of sessions, using PUBLICLY REPORTED facts (attack
duration, records affected, attack mechanism) to parameterize a
behavioral escalation curve.

These are simulated proxies calibrated to reported facts, not the
actual raw telemetry from the breached organizations, which has never
been made public. See paper Section 5.5 and Section 8 for the framing
and limitations of this reconstruction.

Sources for the reported facts used to parameterize each case:
  - Twilio (2022): https://www.twilio.com/blog/august-2022-social-engineering-attack
  - MGM Resorts (2023): https://www.sec.gov/Archives/edgar/data/0000789570/000119312523251667/d461062d8k.htm
  - BenefitMall (2019): https://www.hipaajournal.com/111k-individuals-notified-of-4-month-email-account-compromise/
  - Payroll Pirates (2025): https://blog.checkpoint.com/email-security/payroll-pirates-one-network-hundreds-of-targets/
"""

import numpy as np
import pandas as pd
from ctsf_model import CTSFEngine, THRESHOLDS

# Each case's escalation profile: (n_steps, peak_step, unit)
# n_steps    -- number of time steps simulated (day/hour/session, per case)
# peak_step  -- step at which the attacker's behavior reaches full severity
CASE_PROFILES = {
    "BenefitMall":    dict(n_steps=130, peak_step=25, unit="day"),
    "MGM":            dict(n_steps=72,  peak_step=18, unit="hour"),
    "Twilio":         dict(n_steps=6,   peak_step=1,  unit="session"),
    "PayrollPirates": dict(n_steps=3,   peak_step=1,  unit="session"),
}


def _escalation_curve(n_steps: int, peak_step: int) -> np.ndarray:
    """Ramp from 0 to 1 severity by peak_step, then plateau."""
    x = np.arange(n_steps)
    return np.clip(x / max(peak_step, 1), 0, 1)


def _case_features(name: str, sev: float) -> dict:
    """
    Raw behavioral features for a given case at a given severity level
    (0 = baseline-normal, 1 = full attack severity). Parameter choices
    reflect the qualitative attack pattern reported for each incident;
    see module docstring for sources.
    """
    if name == "BenefitMall":
        return {
            "login_hour": 11 - 8 * sev,           # drifts toward off-hours
            "atypical_actions": 0.3 + 4 * sev,     # sustained search/read behavior
            "records_touched": 6 + 900 * sev,      # bulk mailbox search volume
            "device_geo_dist": min(1.0, 0.2 + sev),
            "peer_dev": 0.1 + 0.75 * sev,
        }
    elif name == "MGM":
        return {
            "login_hour": 11,                        # helpdesk reset during business hours
            "atypical_actions": 0.3 + 5 * sev,        # admin/IdP actions atypical for role
            "records_touched": 6 + 2000 * sev,        # ransomware staging = high volume
            "device_geo_dist": min(1.0, 0.5 + sev),
            "peer_dev": 0.1 + 0.85 * sev,
        }
    elif name == "Twilio":
        return {
            "login_hour": 11,
            "atypical_actions": 0.3 + 3 * sev,
            "records_touched": 6 + 40 * sev,
            "device_geo_dist": 0.9,                   # unregistered attacker device
            "peer_dev": 0.1 + 0.6 * sev,
        }
    elif name == "PayrollPirates":
        return {
            "login_hour": 11,
            "atypical_actions": 0.3 + 5 * sev,         # jump straight to bank-detail edit
            "records_touched": 6 + 15 * sev,
            "device_geo_dist": 0.9,
            "peer_dev": 0.1 + 0.8 * sev,
        }
    else:
        raise ValueError(f"Unknown case: {name}")


def reconstruct_case(engine: CTSFEngine, name: str, n_steps: int,
                      peak_step: int) -> tuple[pd.DataFrame, int | None]:
    """
    Run the CTSF engine over a reconstructed timeline for one case.

    Returns (trajectory_df, detect_step) where detect_step is the first
    time step at which TS(t) fell below the termination/lock threshold
    (0.40), or None if it never did.
    """
    ramp = _escalation_curve(n_steps, peak_step)
    rows = [_case_features(name, sev) for sev in ramp]
    df = pd.DataFrame(rows)

    results = []
    for t, row in df.iterrows():
        # Continuous daily/session anomalous behavior => negligible idle gap
        TS, s, A = engine.compute_TS(row, last_anomaly_gap=0.0)
        results.append({"t": t, "TS": TS, "A": A,
                         **{f"s_{k}": v for k, v in s.items()}})
    res = pd.DataFrame(results)

    detect_idx = res.index[res["TS"] < THRESHOLDS["read_only"]]
    detect_t = int(detect_idx[0]) if len(detect_idx) else None
    return res, detect_t


def run_all_cases(engine: CTSFEngine) -> pd.DataFrame:
    """Run all four case reconstructions and return a summary DataFrame."""
    summary_rows = []
    for name, params in CASE_PROFILES.items():
        res, detect_t = reconstruct_case(engine, name, params["n_steps"],
                                          params["peak_step"])
        res.to_csv(f"results/{name}_trajectory.csv", index=False)
        summary_rows.append({
            "case": name,
            "unit": params["unit"],
            "total_steps_simulated": params["n_steps"],
            "ctsf_detection_point_simulated": detect_t,
            "TS_at_start": round(res["TS"].iloc[0], 3),
            "TS_at_final_step": round(res["TS"].iloc[-1], 3),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv("results/case_summary.csv", index=False)
    return summary
