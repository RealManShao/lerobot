#!/usr/bin/env python

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PATH = Path("docs/source/assets/drifting/0724_inference_latency.png")

STATISTICS = ("P50", "P99", "Maximum", "Mean")
COMPONENTS = {
    "Three-camera decode": (6.977, 8.001, 8.025, 7.080),
    "Observation preparation": (3.827, 4.301, 4.325, 3.851),
    "Policy preprocessing": (13.016, 17.004, 17.956, 13.319),
    "Model input preparation": (0.285, 0.382, 0.389, 0.290),
    "VLM backbone": (42.549, 74.196, 83.696, 44.544),
    "Drifting head": (13.608, 16.383, 16.545, 13.943),
    "Action postprocessing": (0.316, 0.414, 0.437, 0.320),
}
END_TO_END = (81.033, 116.262, 123.771, 83.486)
COLORS = (
    "#4C78A8",
    "#72B7B2",
    "#F2CF5B",
    "#B8B8B8",
    "#E45756",
    "#7A5195",
    "#54A24B",
)


def main() -> None:
    output_path = Path.cwd() / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 9.5), constrained_layout=True)
    positions = np.arange(len(STATISTICS))
    bottoms = np.zeros(len(STATISTICS))

    for (label, values), color in zip(COMPONENTS.items(), COLORS, strict=True):
        values_array = np.asarray(values)
        axis.bar(
            positions,
            values_array,
            width=0.64,
            bottom=bottoms,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        for position, bottom, value in zip(positions, bottoms, values_array, strict=True):
            if value >= 3.0:
                axis.text(
                    position,
                    bottom + value / 2,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if color in {"#4C78A8", "#E45756", "#7A5195"} else "#202124",
                )
        bottoms += values_array

    for position, value, bar_top in zip(positions, END_TO_END, bottoms, strict=True):
        axis.annotate(
            f"E2E {value:.1f} ms",
            (position, bar_top),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    axis.set_title("Drifting Full-Pipeline Inference Latency", fontsize=16, pad=34)
    axis.set_subtitle = None
    axis.text(
        0,
        1.015,
        "Three 640x480 AV1 camera views, A100 80GB, warm cache, n=30",
        transform=axis.transAxes,
        fontsize=10,
        color="#4A4A4A",
    )
    axis.set_ylabel("Latency (ms)")
    axis.set_xticks(positions, STATISTICS)
    axis.set_ylim(0, max(bottoms.max(), max(END_TO_END)) * 1.18)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
    )

    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(output_path)


if __name__ == "__main__":
    main()