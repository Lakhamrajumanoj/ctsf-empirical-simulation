"""
plot_trajectories.py

Generates the trust-score trajectory figure (paper Figure 6), plotting
each reconstructed case's TS(t) curve against the three response
threshold bands defined in Section 4.4.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from case_reconstructions import CASE_PROFILES


def plot_all(output_path: str = "figures/ctsf_trust_trajectories.png"):
    cases = list(CASE_PROFILES.keys())
    units = {name: params["unit"] for name, params in CASE_PROFILES.items()}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, name in zip(axes.flat, cases):
        df = pd.read_csv(f"results/{name}_trajectory.csv")
        ax.plot(df["t"], df["TS"], color="#1f77b4", linewidth=2)
        ax.axhline(0.80, color="green", linestyle="--", linewidth=1,
                   label="Full access (>=0.80)")
        ax.axhline(0.60, color="orange", linestyle="--", linewidth=1,
                   label="Re-auth (0.60-0.80)")
        ax.axhline(0.40, color="red", linestyle="--", linewidth=1,
                   label="Terminate (<0.40)")
        ax.set_title(f"{name} ({units[name]}-level reconstruction)")
        ax.set_xlabel(units[name])
        ax.set_ylabel("Trust Score TS(t)")
        ax.set_ylim(0, 1.05)

    fig.suptitle("CTSF Trust Score Trajectories: Simulated Reconstructions "
                 "of Four Documented Incidents", fontsize=13)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    plot_all()
