#!/usr/bin/env python
"""Aggregate LIBERO latency-benchmark ``summary.json`` files into one comparison plot.

Reads the per-task ``summary.json`` files produced by
``benchmark_libero_latency.py`` (one per LIBERO task suite, each containing a
``{policy: {backbone_ms, action_head_ms, success_rate, ...}}`` mapping) and
renders a single grouped bar chart:
  - x-axis is grouped by task suite (libero_spatial, libero_object, ...)
  - within each group, one stacked bar (backbone + action-head) per policy,
    each policy using its own color theme
  - success-rate delta between the best and worst policy is annotated above each group

Usage
-----
    python scripts/latency_test/plot_libero_summary.py \
        --stats-dir Experiment-result/LIBERO_latency_stats \
        --output outputs/latency_bench/libero_summary.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Fixed policy order + color theme (dark = backbone, light = action head)
_POLICY_THEMES: dict[str, dict[str, str]] = {
    "Xihe666/pi05_libero_full_2A800_bs32_20k": {"backbone": "#38761D", "head": "#93C47D"},
    "Xihe666/gr00t_n17_libero": {"backbone": "#1F4E79", "head": "#6FA8DC"},
    "Xihe666/drif_ov_libero0809": {"backbone": "#B45F06", "head": "#F6B26B"},
}

# Preferred left-to-right task ordering; unknown tasks are appended alphabetically
_TASK_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]


def _load_task_summaries(stats_dir: Path) -> dict[str, dict]:
    summaries = {}
    for task_dir in sorted(stats_dir.iterdir()):
        summary_path = task_dir / "summary.json"
        if task_dir.is_dir() and summary_path.exists():
            summaries[task_dir.name] = json.loads(summary_path.read_text())
    return summaries


def _ordered_tasks(task_names: list[str]) -> list[str]:
    ordered = [t for t in _TASK_ORDER if t in task_names]
    ordered += sorted(t for t in task_names if t not in _TASK_ORDER)
    return ordered


def _plot(summaries: dict[str, dict], policies: list[str], output_path: Path) -> None:
    tasks = _ordered_tasks(list(summaries.keys()))
    n_tasks = len(tasks)
    n_policies = len(policies)

    group_width = 0.7
    bar_width = group_width / n_policies
    x = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100, constrained_layout=True)
    ax_success = ax.twinx()

    for p_idx, policy in enumerate(policies):
        theme = _POLICY_THEMES.get(policy, {"backbone": "#666666", "head": "#BBBBBB"})
        offset = (p_idx - (n_policies - 1) / 2) * bar_width
        bx = x + offset

        backbone_vals = np.array([summaries[t].get(policy, {}).get("backbone_ms", 0.0) for t in tasks])
        head_vals = np.array([summaries[t].get(policy, {}).get("action_head_ms", 0.0) for t in tasks])
        success_vals = np.array([summaries[t].get(policy, {}).get("success_rate", np.nan) for t in tasks])
        total_vals = backbone_vals + head_vals

        label = policy.rsplit("/", 1)[-1]
        ax.bar(bx, backbone_vals, width=bar_width * 0.92, color=theme["backbone"], label=f"{label} · backbone")
        ax.bar(bx, head_vals, width=bar_width * 0.92, bottom=backbone_vals, color=theme["head"], label=f"{label} · head")

        for xi, backbone, head, total in zip(bx, backbone_vals, head_vals, total_vals):
            if backbone >= 3.0:
                ax.text(xi, backbone / 2, f"{backbone:.1f}", ha="center", va="center", color="white", fontsize=8)
            if head >= 3.0:
                ax.text(xi, backbone + head / 2, f"{head:.1f}", ha="center", va="center", color="black", fontsize=8)
            if total > 0:
                ax.text(xi, total + 1.2, f"{total:.1f} ms", ha="center", va="bottom", fontsize=8)

        ax_success.plot(
            bx, success_vals, marker="D", markersize=7, linestyle="none",
            color=theme["backbone"], markeredgecolor="white", markeredgewidth=1, zorder=5,
        )
        for xi, succ in zip(bx, success_vals):
            if not np.isnan(succ):
                ax_success.text(
                    xi, succ + 3, f"{succ:.0f}%", ha="center", va="bottom",
                    fontsize=8, color=theme["backbone"], fontweight="bold", zorder=5,
                )

    # Success-rate delta annotation: bracket between gr00t and drif_ov only, "-|-" style
    _delta_a, _delta_b = "Xihe666/gr00t_n17_libero", "Xihe666/drif_ov_libero0809"
    if _delta_a in policies and _delta_b in policies:
        bracket_x_offset = bar_width * ((n_policies - 1) / 2 + 0.75)
        for xi, task in zip(x, tasks):
            succ_a = summaries[task].get(_delta_a, {}).get("success_rate", np.nan)
            succ_b = summaries[task].get(_delta_b, {}).get("success_rate", np.nan)
            if np.isnan(succ_a) or np.isnan(succ_b):
                continue
            top, bottom = max(succ_a, succ_b), min(succ_a, succ_b)
            delta = top - bottom
            color = "#2E7D32"
            bx = xi + bracket_x_offset
            cap = 0.045 * bar_width
            ax_success.plot(
                [bx - cap, bx + cap, bx + cap, bx + cap, bx - cap],
                [bottom, bottom, bottom, top, top],
                color=color, linewidth=1.4, zorder=6, solid_capstyle="butt",
            )
            ax_success.text(
                bx + cap * 1.8, (top + bottom) / 2, f"Δ {delta:.0f}%",
                ha="left", va="center", fontsize=8, fontweight="bold", color=color,
            )

    ax.set_ylabel("Inference Time per Call (ms)")
    ax.set_xticks(x, [t.replace("libero_", "").replace("_", " ").title() for t in tasks])
    ax.set_xlabel("LIBERO Task Suite")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines[["top"]].set_visible(False)
    ax.set_ylim(0, max(float(ax.get_ylim()[1]), 30) + 15)

    ax_success.set_ylabel("Success Rate (%) ◆")
    ax_success.set_ylim(0, 115)
    ax_success.spines[["top"]].set_visible(False)

    ax.set_title("LIBERO Inference Latency and Success Rate", fontsize=12, pad=14)
    ax.legend(loc="upper left", fontsize=8, ncol=1)

    fig.savefig(output_path, dpi=100, facecolor="white")
    print(f"[plot] saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats-dir", default="Experiment-result/LIBERO_latency_stats", metavar="DIR")
    parser.add_argument("--output", default="outputs/latency_bench/libero_summary.png", metavar="PNG")
    parser.add_argument(
        "--policies", nargs="+", default=list(_POLICY_THEMES.keys()), metavar="HF_REPO",
        help="Policies to include, in display order (default: pi05_libero_full_2A800_bs32_20k, "
        "gr00t_n17_libero, drif_ov_libero0809).",
    )
    args = parser.parse_args()

    stats_dir = Path(args.stats_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = _load_task_summaries(stats_dir)
    if not summaries:
        raise SystemExit(f"No summary.json files found under {stats_dir}")

    _plot(summaries, args.policies, output_path)


if __name__ == "__main__":
    main()
