"""
Per-domain failure breakdown on the position-split held-out set.

Splits the held-out predictions (results/held_out_predictions_position_split.csv,
produced by 04_run_experiments.py) by SpCas9 structural domain and reports
Spearman rho per domain per model, to see WHERE the sequence model's signal
holds up and where it collapses, rather than only one pooled number.

Domain boundaries (1-indexed, inclusive, full-length 1368-aa SpCas9 numbering)
are the canonical ones from the Cas9-sgRNA-DNA crystal structure papers
(Nishimasu et al. 2014, Cell, PDB 4CMP; Anders et al. 2014, Nature, PDB 4UN3 --
the same structure used for this project's structural features):
  RuvC-I:        1-59
  Bridge helix: 60-93     (grouped into "Other" below -- short linker, not
                            one of the four domains the assignment/report asks about)
  REC lobe:     94-718    (REC1+REC2+REC3, treated as one bucket)
  RuvC-II:      719-775
  HNH:          776-908
  RuvC-III:     909-1099
  PI:           1100-1368 (PAM-interacting / WED / CTD region)
These are approximate consensus boundaries from the literature, not re-derived
here; treat this as a coarse structural grouping, not a precise domain model.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

DOMAIN_RANGES = [
    ("RuvC", [(1, 59), (719, 775), (909, 1099)]),
    ("Bridge/Other", [(60, 93)]),
    ("REC", [(94, 718)]),
    ("HNH", [(776, 908)]),
    ("PI", [(1100, 1368)]),
]

MODEL_COLS = [
    "pred_ridge_embedding",
    "pred_ridge_embedding_struct",
    "pred_esm_zero_shot",
    "pred_blosum62",
]


def assign_domain(position):
    for name, ranges in DOMAIN_RANGES:
        for lo, hi in ranges:
            if lo <= position <= hi:
                return name
    return "Unassigned"


def spearman(y_true, y_pred):
    if len(y_true) < 3:
        return None
    r, _ = spearmanr(y_true, y_pred)
    return None if np.isnan(r) else float(r)


def main():
    df = pd.read_csv(f"{RESULTS_DIR}/held_out_predictions_position_split.csv")
    df["domain"] = df["position"].apply(assign_domain)

    n_positions_total = {name: sum(hi - lo + 1 for lo, hi in ranges) for name, ranges in DOMAIN_RANGES}

    report = {}
    print(f"{'domain':<14}{'n_test_mutants':>15}{'n_test_positions':>18}   " +
          "  ".join(f"{c.replace('pred_', ''):>22}" for c in MODEL_COLS))
    for name, _ in DOMAIN_RANGES:
        sub = df[df["domain"] == name]
        n_pos = sub["position"].nunique()
        row = {"n_test_mutants": int(len(sub)), "n_test_positions": int(n_pos),
               "n_domain_positions_total": n_positions_total[name]}
        vals = []
        for col in MODEL_COLS:
            rho = spearman(sub["DMS_score_true"].values, sub[col].values)
            row[col] = rho
            vals.append("n/a" if rho is None else f"{rho:.3f}")
        report[name] = row
        print(f"{name:<14}{len(sub):>15}{n_pos:>18}   " + "  ".join(f"{v:>22}" for v in vals))

    with open(f"{RESULTS_DIR}/domain_breakdown.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {RESULTS_DIR}/domain_breakdown.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        domains = [name for name, _ in DOMAIN_RANGES]
        x = np.arange(len(domains))
        width = 0.2
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, col in enumerate(MODEL_COLS):
            vals = [report[d][col] if report[d][col] is not None else 0.0 for d in domains]
            ax.bar(x + (i - 1.5) * width, vals, width, label=col.replace("pred_", ""))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(domains)
        ax.set_ylabel("Spearman rho (position-split test)")
        ax.set_title("Held-out rank correlation by SpCas9 domain")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{FIGURES_DIR}/domain_breakdown.png", dpi=150)
        print(f"Saved {FIGURES_DIR}/domain_breakdown.png")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
