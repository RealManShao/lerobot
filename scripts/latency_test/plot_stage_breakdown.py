#!/usr/bin/env python
"""Full per-stage inference latency comparison across policies.

Produces a side-by-side stacked bar chart (P50 / P99 / Maximum / Mean) showing
every pipeline stage for Drifting, Drif-OV, and GR00T-N1.7.

Data is embedded directly as constants.  After running
``scripts/latency_test/benchmark_offline_latency.py`` for each policy, update the
corresponding entry in ``DATA`` and re-run this script to regenerate the plot.

Usage
-----
    python scripts/latency_test/plot_stage_breakdown.py

Output
------
    docs/source/assets/inference_latency_stage_breakdown.png

Data sources
------------
- Drifting   : benchmark_offline_latency.py --policy drifting
               (Tron2 / siemens-v3-disturb, A100 80GB, 3×640×480 AV1, warm cache, n=30, 2026-07-24)
- Drif-OV    : benchmark_offline_latency.py --policy drif_ov
               (Panda / Libero-10, A100 80GB, 2×cam, warm cache, n=30 — TBD)
- GR00T-N1.7 : benchmark_offline_latency.py --policy groot_n17
               (Xihe666/gr00t_n17_libero_1A800_0811, Panda / Libero-10, A100 80GB — TBD)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUTPUT_PATH = Path("docs/source/assets/inference_latency_stage_breakdown.png")

STATISTICS = ("P50", "P99", "Maximum", "Mean")

# ── Per-policy stage data (P50, P99, Maximum, Mean) in milliseconds ──────────
# Sections marked "# TBD" must be filled in after running the benchmark.
DATA: dict[str, dict] = {
    "Drifting": {
        "subtitle": "Tron2 · 3×640×480 AV1 · A100 80GB · n=30 · 2026-07-24",
        "components": {
            "Three-camera decode":  (6.977,  8.001,  8.025,  7.080),
            "Observation prep":     (3.827,  4.301,  4.325,  3.851),
            "Policy preprocessor":  (13.016, 17.004, 17.956, 13.319),
            "Model input prep":     (0.285,  0.382,  0.389,  0.290),
            "VLM backbone":         (42.549, 74.196, 83.696, 44.544),
            "Action head":          (13.608, 16.383, 16.545, 13.943),
            "Action postprocessor": (0.316,  0.414,  0.437,  0.320),
        },
        "end_to_end": (81.033, 116.262, 123.771, 83.486),
    },
    "Drif-OV": {  # TBD — run: benchmark_offline_latency.py --policy drif_ov
        "subtitle": "Panda / Libero-10 · 2×cam · A100 80GB · TBD",
        "components": {
            "Two-camera decode":    (0.0, 0.0, 0.0, 0.0),
            "Observation prep":     (0.0, 0.0, 0.0, 0.0),
            "Policy preprocessor":  (0.0, 0.0, 0.0, 0.0),
            "Model input prep":     (0.0, 0.0, 0.0, 0.0),
            "VLM backbone":         (0.0, 0.0, 0.0, 0.0),
            "Action head":          (0.0, 0.0, 0.0, 0.0),
            "Action postprocessor": (0.0, 0.0, 0.0, 0.0),
        },
        "end_to_end": (0.0, 0.0, 0.0, 0.0),
    },
    "GR00T-N1.7\n(libero_1A800)": {  # TBD — run: benchmark_offline_latency.py --policy groot_n17
        "subtitle": "Panda / Libero-10 · 2×cam · A100 80GB · TBD",
        "components": {
            "Two-camera decode":    (0.0, 0.0, 0.0, 0.0),
            "Observation prep":     (0.0, 0.0, 0.0, 0.0),
            "Policy preprocessor":  (0.0, 0.0, 0.0, 0.0),
            "Model input prep":     (0.0, 0.0, 0.0, 0.0),
            "VLM backbone":         (0.0, 0.0, 0.0, 0.0),
            "Action head":          (0.0, 0.0, 0.0, 0.0),
            "Action postprocessor": (0.0, 0.0, 0.0, 0.0),
        },
        "end_to_end": (0.0, 0.0, 0.0, 0.0),
    },
}

COMPONENT_COLORS: dict[str, str] = {
    "Three-camera decode":  "#4C78A8",
    "Two-camera decode":    "#4C78A8",
    "Observation prep":     "#72B7B2",
    "Policy preprocessor":  "#F2CF5B",
    "Model input prep":     "#B8B8B8",
    "VLM backbone":         "#E45756",
    "Action head":          "#7A5195",
    "Action postprocessor": "#54A24B",
}
_DARK = {"#4C78A8", "#E45756", "#7A5195"}


def main() -> None:
    output_path = Path.cwd() / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_policies = len(DATA)
    fig, axes = plt.subplots(
        1, n_policies,
        figsize=(6 * n_policies, 9),
        constrained_layout=True,
        sharey=False,
    )
    if n_policies == 1:
        axes = [axes]

    seen_labels: set[str] = set()

    for ax, (policy_name, policy_data) in zip(axes, DATA.items(), strict=True):
        positions = np.arange(len(STATISTICS))
        bottoms = np.zeros(len(STATISTICS))

        for component, values in policy_data["components"].items():
            color = COMPONENT_COLORS.get(component, "#AAAAAA")
            arr = np.asarray(values)
            legend_label = component if component not in seen_labels else "_nolegend_"
            seen_labels.add(component)
            ax.bar(
                positions, arr, width=0.64, bottom=bottoms,
                color=color, edgecolor="white", linewidth=0.6, label=legend_label,
            )
            for pos, bot, val in zip(positions, bottoms, arr, strict=True):
                if val >= 4.0:
                    ax.text(
                        pos, bot + val / 2, f"{val:.1f}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if color in _DARK else "#202124",
                    )
            bottoms += arr

        for pos, e2e_val, bar_top in zip(positions, policy_data["end_to_end"], bottoms, strict=True):
            label_txt = f"E2E {e2e_val:.1f} ms" if e2e_val > 0 else "TBD"
            ax.annotate(
                label_txt, (pos, bar_top),
                xytext=(0, 7), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

        ax.set_title(policy_name, fontsize=13, fontweight="bold", pad=28)
        ax.text(
            0.5, 1.015, policy_data["subtitle"],
            transform=ax.transAxes, fontsize=8, color="#4A4A4A", ha="center",
        )
        ax.set_ylabel("Latency (ms)")
        ax.set_xticks(positions, STATISTICS, fontsize=9)
        max_height = max(bottoms.max(), max(policy_data["end_to_end"]))
        ax.set_ylim(0, max(max_height * 1.22, 10))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[-1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(), unique.keys(),
        loc="lower center", ncol=4, frameon=False, fontsize=8.5,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle(
        "Full-Pipeline Inference Latency — Stage Breakdown",
        fontsize=15, fontweight="bold", y=1.02,
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(output_path)


if __name__ == "__main__":
    main()
