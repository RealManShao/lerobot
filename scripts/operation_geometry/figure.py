"""Render the sign-pair test and held-out energy errors from the geometry scores."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scores", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    rows = json.loads(args.scores.read_text())
    fig, (scatter, bars) = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    joint = [r for r in rows if r["outcome"] == "joint"]
    pairs = np.array([pair for r in joint for pair in r["sign_pairs"]])
    displacement = 1e3 * 0.02 * np.sqrt(pairs)
    visible = displacement.min(axis=1) > 0
    x, y = displacement[visible].T
    lo, hi = min(x.min(), y.min()) * 0.75, max(x.max(), y.max()) * 1.3
    scatter.plot([lo, hi], [lo, hi], color="0.55", ls="--", lw=1)
    scatter.scatter(x, y, s=16, color="#156b82", alpha=0.63, edgecolors="none")
    scatter.set(
        xscale="log",
        yscale="log",
        xlim=(lo, hi),
        ylim=(lo, hi),
        xlabel="Same-sign commands (mm)",
        ylabel="Alternating-sign commands (mm)",
        title="Equal coordinate energies, different outcomes",
    )
    names = ["expert", "expert_blocks", "physics_rank3", "physics_full"]
    idx = np.arange(len(names))
    for outcome, off, color, label in [
        ("joint", -0.19, "#156b82", "Hand + objects (60 states)"),
        ("object", 0.19, "#c26030", "Objects only (42 active states)"),
    ]:
        active = [r for r in rows if r["outcome"] == outcome and r["reference_movement_um"] >= 1]
        vals = [np.median([r["tests"]["0.04"][m]["energy_nrmse"] for r in active]) for m in names]
        bars.bar(idx + off, vals, width=0.36, color=color, label=label)
    bars.set_xticks(idx, ["IDP", "IDP +\nblock means", "Rank-3\nphysics", "Full\nphysics"])
    bars.set(
        ylabel="Median normalized RMSE (lower is better)", title="Energy on unseen perturbation directions"
    )
    bars.legend(frameon=False, fontsize=9)
    for ax in (scatter, bars):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
