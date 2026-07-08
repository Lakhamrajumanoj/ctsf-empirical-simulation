"""
ctsf_model.py

Core implementation of the Cognitive Trust Scoring Framework (CTSF)
Layer 3 behavioral trust engine, as defined in:

    Lakhamraju, M. V. "Securing the Digital Core: A Cognitive Trust
    Scoring Framework for ERP System Security."

This module implements:
  - Synthetic baseline ("normal") session generation per ERP role
  - Five Isolation Forest anomaly detectors, one per signal dimension
  - The weighted aggregation formula A(t) = sum(w_i * s_i(t))
  - The temporal decay factor alpha(t) = exp(-lambda * delta_t)
  - The trust score TS(t) = 1 - alpha(t) * A(t)

IMPORTANT: All session data used here is SYNTHETIC, generated to
reflect plausible role-based behavior patterns. It is NOT real
telemetry from any production ERP system or from the breached
organizations discussed in the paper's case studies. This code
supports a simulation-based reconstruction calibrated to publicly
reported incident facts, not a live-system evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# ----------------------------------------------------------------------
# Weights and thresholds as defined in the paper (Section 4.4)
# ----------------------------------------------------------------------
WEIGHTS = {
    "temporal": 0.15,
    "behavioural": 0.20,
    "volumetric": 0.25,
    "geo_device": 0.25,
    "role_norm": 0.15,
}

THRESHOLDS = {
    "full_access": 0.80,   # TS >= 0.80 -> full access granted
    "reauth": 0.60,        # 0.60 <= TS < 0.80 -> re-authentication challenge
    "read_only": 0.40,      # 0.40 <= TS < 0.60 -> session demoted to read-only
    # TS < 0.40 -> session terminated, security team alerted
}

# Feature used to compute each signal's anomaly sub-score
FEATURE_MAP = {
    "temporal": "login_hour",
    "behavioural": "atypical_actions",
    "volumetric": "records_touched",
    "geo_device": "device_geo_dist",
    "role_norm": "peer_dev",
}

DEFAULT_LAMBDA = 1.0  # decay constant (per time step); a modeling
                       # assumption for this simulation, not empirically
                       # fit -- see paper Section 8 (Limitations)


def alpha(delta_t: float, lam: float = DEFAULT_LAMBDA) -> float:
    """Temporal decay factor: alpha(t) = exp(-lambda * delta_t)."""
    return float(np.exp(-lam * delta_t))


def gen_baseline_sessions(role: str = "payroll_analyst", n: int = 800) -> pd.DataFrame:
    """
    Generate n synthetic 'normal' sessions for a given ERP role.

    Uses the global numpy random state (seeded once in run_simulation.py)
    rather than a per-call generator, so that repeated calls draw fresh,
    non-overlapping samples from a single continuous stream -- this
    matters for reproducing the exact figures reported in the paper and
    for avoiding train/test overlap between the baseline population and
    the held-out test set.

    Returns a DataFrame with raw behavioral features:
      login_hour        -- hour of day session started (0-23)
      atypical_actions  -- count of action types outside role profile
      records_touched   -- records accessed/modified during session
      device_geo_dist   -- 0 = known device/location, 1 = fully unknown
      peer_dev          -- behavioral distance from peer-group norm (0-1)
    """
    if role == "payroll_analyst":
        login_hour = np.clip(np.random.normal(11, 2.5, n), 0, 23)
        records = np.random.poisson(6, n)
    elif role == "hr_specialist":
        login_hour = np.clip(np.random.normal(10, 2.5, n), 0, 23)
        records = np.random.poisson(8, n)
    else:
        login_hour = np.clip(np.random.normal(10.5, 2.5, n), 0, 23)
        records = np.random.poisson(5, n)

    atypical = np.random.poisson(0.4, n)
    device_geo = np.random.binomial(1, 0.03, n).astype(float)
    peer_dev = np.clip(np.random.normal(0.1, 0.08, n), 0, 1)

    return pd.DataFrame({
        "login_hour": login_hour,
        "atypical_actions": atypical,
        "records_touched": records,
        "device_geo_dist": device_geo,
        "peer_dev": peer_dev,
    })


class CTSFEngine:
    """
    Trains one Isolation Forest per signal dimension on a baseline
    population, then scores new sessions into sub-scores s_i in [0,1]
    and computes the aggregate trust score TS(t).
    """

    def __init__(self, baseline: pd.DataFrame, contamination: float = 0.05,
                 n_estimators: int = 200, seed: int = 42):
        self.forests = {}
        self.score_ranges = {}
        for sig, col in FEATURE_MAP.items():
            X = baseline[[col]].values
            clf = IsolationForest(n_estimators=n_estimators,
                                   contamination=contamination,
                                   random_state=seed).fit(X)
            train_scores = -clf.decision_function(X)
            self.forests[sig] = clf
            self.score_ranges[sig] = (train_scores.min(), train_scores.max())

    def signal_score(self, sig: str, value: float) -> float:
        """Map a raw feature value to an anomaly sub-score in [0,1]."""
        clf = self.forests[sig]
        raw = -clf.decision_function(np.array([[value]]))[0]
        lo, hi = self.score_ranges[sig]
        return float(np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1))

    def compute_TS(self, session_row: pd.Series, last_anomaly_gap: float = 0.0,
                   lam: float = DEFAULT_LAMBDA):
        """
        Compute the trust score for a single session.

        Returns (TS, sub_scores_dict, A) where:
          TS          -- final trust score in [0,1]
          sub_scores  -- dict of s_i values per signal dimension
          A           -- weighted aggregate anomaly score
        """
        s = {sig: self.signal_score(sig, session_row[FEATURE_MAP[sig]])
             for sig in FEATURE_MAP}
        A = sum(WEIGHTS[sig] * s[sig] for sig in FEATURE_MAP)
        a = alpha(last_anomaly_gap, lam)
        TS = 1 - a * A
        return TS, s, A

    def classify(self, TS: float) -> str:
        """Map a trust score to the corresponding CTSF response action."""
        if TS >= THRESHOLDS["full_access"]:
            return "full_access"
        elif TS >= THRESHOLDS["reauth"]:
            return "reauth_challenge"
        elif TS >= THRESHOLDS["read_only"]:
            return "read_only"
        else:
            return "terminate_and_alert"
