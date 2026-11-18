"""Draw results/f1_summary.png from results/metrics.json.

Usage:  python scripts/plot_results.py   (needs matplotlib, see requirements-dev.txt)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
m = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
cls, rob = m["classification"], m["robustness"]["global"]

LEAKY, HONEST, BASELINE = "#eb6834", "#2a78d6", "#a3a29c"
TEXT, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"

rows = [  # (label, F1, color) top to bottom
    ("Original C++ setup, with leakage\n(converging optimiser)", cls["cpp_replica_leaky"]["models"]["logreg_lbfgs"]["f1"], LEAKY),
    ("Original C++ setup, with leakage\n(SGD exactly as in C++)", cls["cpp_replica_leaky"]["models"]["logreg_sgd_as_cpp"]["f1"], LEAKY),
    ("Original C++ setup, leak removed", cls["no_leakage_cpp_setup"]["models"]["logreg_lbfgs"]["f1"], HONEST),
    ("Clean pipeline, random split", rob["cv_random"]["models"]["hist_gradient_boosting"]["f1"], HONEST),
    ("Clean pipeline, unseen stations", rob["cv_by_station"]["models"]["hist_gradient_boosting"]["f1"], HONEST),
    ("Clean pipeline, unseen countries", rob["cv_by_country"]["models"]["hist_gradient_boosting"]["f1"], HONEST),
    ("Baseline: pollutant + unit only", rob["baseline_pollutant_unit_only"]["models"]["logreg"]["f1"], BASELINE),
]

fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
y = range(len(rows))[::-1]
ax.barh(list(y), [r[1] for r in rows], color=[r[2] for r in rows], height=0.62, edgecolor=SURFACE, linewidth=2)
for yi, (_, f1, _) in zip(y, rows):
    ax.text(f1 + 0.012, yi, f"{f1:.2f}", va="center", fontsize=9, color=TEXT)

ax.set_yticks(list(y), [r[0] for r in rows], fontsize=8.5, color=TEXT)
ax.set_xlim(0, 1.08)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0)
ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ax.spines.values():
    side.set_visible(False)
ax.set_xlabel("F1 score, \"high pollution\" class (higher is better)", fontsize=9, color=MUTED)
ax.set_title("A leaked feature makes the model look perfect; honest evaluation does not",
             fontsize=11, color=TEXT, loc="left", pad=26)
ax.legend(
    handles=[Patch(color=LEAKY, label="Target leaked into features"),
             Patch(color=HONEST, label="No leakage"),
             Patch(color=BASELINE, label="Naive baseline")],
    loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.5,
    labelcolor=TEXT, borderaxespad=0.2, handlelength=1.2,
)
fig.text(0.99, 0.01, "Gradient boosting except the first three rows (logistic regression). Clean rows: 5-fold CV.",
         ha="right", fontsize=7, color=MUTED)
fig.tight_layout()
out = ROOT / "results" / "f1_summary.png"
fig.savefig(out, facecolor=SURFACE)
print("Written", out)
