"""
classifier_validation.py

Evaluates the CTSF trust-score classifier on a held-out synthetic test
set of labeled normal vs. attack-pattern sessions, reporting precision,
recall, F1-score, and false positive rate. These are the self-computed
figures reported in paper Section 5.5, distinct from the
literature-derived estimates in Table 3.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from ctsf_model import CTSFEngine, gen_baseline_sessions, THRESHOLDS


def build_test_set(n_normal: int = 400, n_attack: int = 100) -> pd.DataFrame:
    """
    Build a labeled test set: 0 = normal session, 1 = attack-pattern session.

    Draws from the same continuous global random stream as the baseline
    training population (see gen_baseline_sessions), so this must be
    called AFTER the baseline population used to train the engine, and
    only once per run, to match the paper's reported figures.
    """
    test_normal = gen_baseline_sessions("payroll_analyst", n_normal)
    test_normal["label"] = 0

    sev = np.random.uniform(0.3, 1.0, n_attack)
    test_attack = pd.DataFrame({
        "login_hour": np.clip(11 - 8 * sev + np.random.normal(0, 1, n_attack), 0, 23),
        "atypical_actions": 0.3 + 4 * sev + np.random.normal(0, 0.3, n_attack),
        "records_touched": 6 + 300 * sev + np.random.normal(0, 20, n_attack),
        "device_geo_dist": np.clip(0.3 + sev + np.random.normal(0, 0.05, n_attack), 0, 1),
        "peer_dev": np.clip(0.1 + 0.7 * sev + np.random.normal(0, 0.05, n_attack), 0, 1),
    })
    test_attack["label"] = 1

    return pd.concat([test_normal, test_attack], ignore_index=True)


def evaluate(engine: CTSFEngine, test: pd.DataFrame,
             flag_threshold: float = THRESHOLDS["reauth"]) -> dict:
    """
    Score every session in the test set and compute classification
    metrics, flagging a session as "detected" if TS < flag_threshold.
    """
    TS_vals = [engine.compute_TS(row, last_anomaly_gap=0.0)[0]
               for _, row in test.iterrows()]
    test = test.copy()
    test["TS"] = TS_vals
    test["pred"] = (test["TS"] < flag_threshold).astype(int)

    prec = precision_score(test["label"], test["pred"])
    rec = recall_score(test["label"], test["pred"])
    f1 = f1_score(test["label"], test["pred"])
    fpr = ((test["pred"] == 1) & (test["label"] == 0)).sum() / (test["label"] == 0).sum()

    test.to_csv("results/classifier_validation.csv", index=False)
    return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "n": len(test)}
