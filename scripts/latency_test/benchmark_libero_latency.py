#!/usr/bin/env python
"""Live LIBERO eval benchmark: measures per-step VLM backbone vs action-head latency.

Runs ``lerobot-eval`` for each policy with timing hooks enabled via the
``LEROBOT_PROFILE_INFERENCE_TIMINGS`` env variable.  Timing hooks are implemented
in ``DriftingN17.get_action``, ``DrifOvN17.get_action``, and ``GR00TN17.get_action``;
each hook writes one CSV row per inference call.

After all eval runs complete the script reads the per-policy CSVs, computes means,
parses the final ``pc_success`` from the eval log, and saves:
  - ``<output_dir>/<task>/summary.json``           aggregated means + success rates
  - ``<output_dir>/<task>/<task>_vlm_vs_head.png`` stacked bar chart

Usage
-----
    # Three-policy comparison on libero_10:
    python scripts/latency_test/benchmark_libero_latency.py \\
        --policies Xihe666/drifting_libero_full \\
                   nvidia/gr00t17-lerobot-libero_10-640 \\
                   Xihe666/drif_ov_libero0809 \\
        --task libero_10 --n-episodes 1

    # Spatial suite with stable averages:
    python scripts/latency_test/benchmark_libero_latency.py \
        --policies Xihe666/gr00t_n17_libero \
        --task libero_spatial --n-episodes 5 --output-dir outputs/bench_new_groot \
        --policy.pretrained_revision 020000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Camera key rename applied to every policy for LIBERO evaluation
_DEFAULT_RENAME_MAP: str = (
    '{"observation.images.image2": "observation.images.wrist_image"}'
)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _is_progress_line(line: str) -> bool:
    """Return True for noisy tqdm/progress output lines."""
    clean = _ANSI_ESCAPE_RE.sub("", line)
    if "\r" in line:
        return True
    if "running_success_rate=" in clean:
        return True
    if "Stepping through eval batches:" in clean:
        return True
    if "Running rollout with at most" in clean:
        return True
    return bool(re.search(r"\b\d+%\|", clean) and ("it/s" in clean or "s/it" in clean))


# ── Evaluation runner ─────────────────────────────────────────────────────────

def _run_eval(
    *,
    policy_path: str,
    policy_pretrained_revision: str | None,
    task: str,
    n_episodes: int,
    csv_path: Path,
    log_path: Path,
    rename_map: str | None,
    extra_env: dict | None,
) -> dict:
    env = {**os.environ, "LEROBOT_PROFILE_INFERENCE_TIMINGS": str(csv_path)}
    if extra_env:
        env.update(extra_env)

    if rename_map is None:
        rename_map = _DEFAULT_RENAME_MAP

    cmd = [
        "lerobot-eval",
        f"--policy.path={policy_path}",
        "--env.type=libero",
        f"--env.task={task}",
        "--eval.batch_size=1",
        f"--eval.n_episodes={n_episodes}",
        "--env.max_parallel_tasks=1",
    ]
    if policy_pretrained_revision:
        cmd.append(f"--policy.pretrained_revision={policy_pretrained_revision}")
    if rename_map:
        cmd.append(f"--rename_map={rename_map}")

    print(f"[bench] {policy_path}  task={task}  n_episodes={n_episodes}")
    print(f"[bench] log → {log_path}")
    with log_path.open("w") as fh, subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            if not _is_progress_line(line):
                fh.write(_ANSI_ESCAPE_RE.sub("", line))
        returncode = proc.wait()

    return {"returncode": returncode, "success_rate": _parse_success_rate(log_path)}


def _parse_success_rate(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    pattern = re.compile(r"['\"]?pc_success['\"]?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)")
    for line in reversed(log_path.read_text(errors="replace").splitlines()):
        match = pattern.search(line)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


# ── CSV reader ────────────────────────────────────────────────────────────────

def _read_timing_csv(csv_path: Path) -> tuple[float, float, int]:
    """Return (mean backbone ms, mean action-head ms, sample count)."""
    backbone_vals: list[float] = []
    action_vals: list[float] = []
    if not csv_path.exists():
        print(f"[bench] WARNING: timing CSV not found: {csv_path}", file=sys.stderr)
        return 0.0, 0.0, 0
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                backbone_vals.append(float(row["backbone_ms"]))
                action_vals.append(float(row["action_head_ms"]))
            except (KeyError, ValueError):
                continue
    if not backbone_vals:
        return 0.0, 0.0, 0
    n = len(backbone_vals)
    return sum(backbone_vals) / n, sum(action_vals) / n, n


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot(
    *,
    policies: list[str],
    backbone_ms: np.ndarray,
    action_head_ms: np.ndarray,
    success_rates: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    total_ms = backbone_ms + action_head_ms
    x = np.arange(len(policies))
    tick_labels = [p.rsplit("/", 1)[-1] for p in policies]

    fig, ax = plt.subplots(
        figsize=(max(9, len(policies) * 4), 7), constrained_layout=True
    )
    ax.bar(x, backbone_ms, width=0.6, color="#4E79A7", label="VLM / Backbone")
    ax.bar(x, action_head_ms, width=0.6, bottom=backbone_ms, color="#F28E2B", label="Action Head")

    for i in range(len(policies)):
        if backbone_ms[i] >= 3.0:
            ax.text(
                x[i], backbone_ms[i] / 2, f"{backbone_ms[i]:.1f} ms",
                ha="center", va="center", color="white", fontsize=10,
            )
        if action_head_ms[i] >= 3.0:
            ax.text(
                x[i], backbone_ms[i] + action_head_ms[i] / 2,
                f"{action_head_ms[i]:.1f} ms",
                ha="center", va="center", color="black", fontsize=10,
            )
        annotation = (
            f"Total: {total_ms[i]:.1f} ms\nSuccess: {success_rates[i]:.1f}%"
            if total_ms[i] > 0
            else f"TBD\nSuccess: {success_rates[i]:.1f}%"
        )
        ax.text(
            x[i], total_ms[i] + 1.5, annotation,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title(title, fontsize=15, pad=12)
    ax.set_ylabel("Inference Time per Call (ms)")
    ax.set_xticks(x, tick_labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(float(np.max(total_ms)) + 22, 30))

    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"[bench] plot → {output_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--policies", nargs="+", required=True, metavar="HF_REPO",
        help="One or more HuggingFace policy repo paths.",
    )
    parser.add_argument(
        "--policy.pretrained_revision", dest="policy_pretrained_revision", default=None, metavar="REV",
        help="Optional policy pretrained revision passed to lerobot-eval.",
    )
    parser.add_argument(
        "--task", default="libero_10",
        choices=["libero_10", "libero_spatial", "libero_goal", "libero_object"],
    )
    parser.add_argument(
        "--n-episodes", type=int, default=5, metavar="N",
        help="Episodes per sub-task (default: 5).",
    )
    parser.add_argument(
        "--output-dir", default="outputs/latency_bench", metavar="DIR",
    )
    parser.add_argument(
        "--rename-map", default=None, metavar="JSON",
        help="JSON observation key rename map applied to all policies.",
    )
    parser.add_argument(
        "--pytorch-alloc-conf", default="expandable_segments:True",
        help="Value for PYTORCH_CUDA_ALLOC_CONF (default: expandable_segments:True).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_env = (
        {"PYTORCH_CUDA_ALLOC_CONF": args.pytorch_alloc_conf}
        if args.pytorch_alloc_conf
        else {}
    )

    all_backbone: list[float] = []
    all_action_head: list[float] = []
    all_success: list[float] = []
    results_meta: dict = {}

    for policy in args.policies:
        safe = policy.replace("/", "_")
        csv_path = output_dir / f"{safe}_timings.csv"
        log_path = output_dir / f"{safe}_eval.log"

        meta = _run_eval(
            policy_path=policy,
            policy_pretrained_revision=args.policy_pretrained_revision,
            task=args.task,
            n_episodes=args.n_episodes,
            csv_path=csv_path,
            log_path=log_path,
            rename_map=args.rename_map,
            extra_env=extra_env,
        )
        if meta["returncode"] != 0:
            print(
                f"[bench] WARNING: {policy} exited {meta['returncode']} — check {log_path}",
                file=sys.stderr,
            )

        backbone_ms, action_head_ms, n_calls = _read_timing_csv(csv_path)
        success = meta["success_rate"] if meta["success_rate"] is not None else 0.0
        if meta["success_rate"] is None:
            print(
                f"[bench] WARNING: could not parse pc_success from {log_path}",
                file=sys.stderr,
            )
        print(
            f"[bench] {policy}: backbone={backbone_ms:.1f} ms  "
            f"head={action_head_ms:.1f} ms  total={backbone_ms + action_head_ms:.1f} ms  "
            f"success={success:.1f}%  n={n_calls}"
        )

        all_backbone.append(backbone_ms)
        all_action_head.append(action_head_ms)
        all_success.append(success)
        results_meta[policy] = {
            "backbone_ms": backbone_ms,
            "action_head_ms": action_head_ms,
            "total_ms": backbone_ms + action_head_ms,
            "success_rate": success,
            "n_timing_calls": n_calls,
            "eval_returncode": meta["returncode"],
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results_meta, indent=2))
    print(f"[bench] summary → {summary_path}")

    task_label = args.task.replace("_", "-").title()
    _plot(
        policies=args.policies,
        backbone_ms=np.array(all_backbone),
        action_head_ms=np.array(all_action_head),
        success_rates=np.array(all_success),
        title=f"LIBERO-{task_label} Inference Cost Breakdown (n_episodes={args.n_episodes})",
        output_path=output_dir / f"{args.task}_vlm_vs_head.png",
    )


if __name__ == "__main__":
    main()
