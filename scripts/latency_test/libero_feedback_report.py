#!/usr/bin/env python
"""Aggregate and plot a LIBERO feedback experiment without loading the policy.

    conda run -n lerobot python scripts/latency_test/libero_feedback_report.py \
        --output-dir outputs/libero_feedback

Only completed, atomically saved episode JSON files are read. Missing observations
are represented by null in JSON, empty cells in CSV, and gaps in plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def statistics(values: list[float | None]) -> dict[str, float | int | None]:
    """Describe finite samples; an empty sample never becomes an artificial zero."""
    array = np.asarray([value for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return dict.fromkeys(("mean", "p50", "p95", "p99", "max")) | {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def wilson_interval(successes: int, total: int) -> list[float | None]:
    """Two-sided 95% Wilson interval for the observed episode success fraction."""
    if not total:
        return [None, None]
    z = 1.959963984540054
    fraction = successes / total
    denominator = 1 + z * z / total
    center = (fraction + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total * total))
    radius /= denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _fraction(values: list[bool | None]) -> dict[str, float | int | None]:
    valid = [bool(value) for value in values if value is not None]
    return {
        "count": len(valid),
        "events": sum(valid),
        "fraction": sum(valid) / len(valid) if valid else None,
    }


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    actions = [action for episode in episodes for action in episode["actions"]]
    plans = [plan for episode in episodes for plan in episode["plans"]]
    successes = [episode for episode in episodes if episode["success"]]
    boundaries = [action for action in actions if action["is_boundary"]]
    # The first action has no preceding action. Exclude it from interior samples.
    interior = [action for action in actions if not action["is_boundary"] and action["step"] > 0]

    def action_stats(key: str, rows: list[dict[str, Any]] = actions) -> dict[str, Any]:
        return statistics([row.get(key) for row in rows])

    row: dict[str, Any] = {
        "episodes": len(episodes),
        "successes": len(successes),
        "success_rate": len(successes) / len(episodes) if episodes else None,
        "success_wilson95": wilson_interval(len(successes), len(episodes)),
        "tasks_observed": len({episode["task_id"] for episode in episodes}),
        "actions": len(actions),
        "plans": len(plans),
        "completion_wall_s_success_only": statistics([episode["wall_time_s"] for episode in successes]),
        "episode_wall_s_all": statistics([episode["wall_time_s"] for episode in episodes]),
        "episode_sim_s_all": statistics([episode["sim_time_s"] for episode in episodes]),
        "episode_steps_all": statistics([episode["steps"] for episode in episodes]),
        "observation_age_ms": action_stats("observation_age_ms"),
        "observation_age_steps": action_stats("observation_age_steps"),
        "pipeline_ms": statistics([plan.get("pipeline_ms") for plan in plans]),
        "e2e_first_action_ms": statistics([plan.get("e2e_first_action_ms") for plan in plans]),
        "deadline_missed": _fraction([action.get("deadline_missed") for action in actions]),
        "ready_deadline_missed": _fraction([action.get("ready_deadline_missed") for action in actions]),
        "dispatch_lateness_ms": action_stats("dispatch_lateness_ms"),
        "env_step_ms": action_stats("env_step_ms"),
        "boundary_interval_error_ms": action_stats("boundary_interval_error_ms", boundaries),
        "boundary_interval_abs_error_ms": statistics(
            [
                abs(action["boundary_interval_error_ms"])
                for action in boundaries
                if action.get("boundary_interval_error_ms") is not None
            ]
        ),
    }
    for kind, selected in (("boundary", boundaries), ("interior", interior)):
        row[f"{kind}_transitions"] = len(selected)
        for metric in ("action_delta_l2", "action_accel_l2", "translation_delta_l2", "rotation_delta_l2"):
            row[f"{kind}_{metric}"] = action_stats(metric, selected)
        row[f"{kind}_gripper_changed"] = _fraction([action.get("gripper_changed") for action in selected])
    return row


def _pair_key(episode: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(episode[key] for key in ("task_id", "episode_index", "seed", "init_state_id"))


def _paired_wall_stats(episodes: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    reference = {_pair_key(episode): episode for episode in baseline}
    matched = [
        (episode, reference[_pair_key(episode)]) for episode in episodes if _pair_key(episode) in reference
    ]
    both_success = [(episode, ref) for episode, ref in matched if episode["success"] and ref["success"]]
    return {
        "baseline_k": 40,
        "matched_episodes": len(matched),
        "both_successful_episodes": len(both_success),
        "k_only_successes": sum(episode["success"] and not ref["success"] for episode, ref in matched),
        "baseline_only_successes": sum(ref["success"] and not episode["success"] for episode, ref in matched),
        "wall_delta_s_both_success": statistics(
            [episode["wall_time_s"] - ref["wall_time_s"] for episode, ref in both_success]
        ),
        "wall_ratio_both_success": statistics(
            [
                episode["wall_time_s"] / ref["wall_time_s"]
                for episode, ref in both_success
                if ref["wall_time_s"] > 0
            ]
        ),
    }


def _flatten(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            result.update(_flatten(value, f"{name}_"))
        elif isinstance(value, list):
            result[name] = json.dumps(value, allow_nan=False)
        else:
            result[name] = value
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]], default_fields: tuple[str, ...] = ("k",)) -> None:
    flattened = [_flatten(row) for row in rows]
    fields = list(dict.fromkeys(key for row in flattened for key in row)) or list(default_fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def _series(rows: list[dict[str, Any]], metric: str, field: str = "mean", scale: float = 1) -> np.ndarray:
    values = [row[metric][field] if field else row[metric] for row in rows]
    return np.asarray([np.nan if value is None else scale * value for value in values])


def _prepare_axes(axes: Any, rows: list[dict[str, Any]]) -> None:
    for axis in np.asarray(axes).flat:
        axis.set_xticks(np.arange(len(rows)), [str(row["k"]) for row in rows])
        axis.set_xlabel("Executed actions before replanning (K)")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)


def _save_figure(figure: Any, output_dir: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"{stem}.{extension}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot(summary: dict[str, Any], output_dir: Path) -> None:
    rows = summary["by_k"]
    if not rows:
        return
    x = np.arange(len(rows))
    coverage = summary["coverage"]
    protocol = summary["protocol"]
    coverage_label = "COMPLETE REQUESTED RUN" if coverage["complete"] else "PARTIAL DATA"
    limit_label = (
        f"SMOKE: shortened episode limit={protocol['max_steps']}"
        if protocol["shortened_episode_limit"]
        else f"episode limit={protocol['max_steps']}"
    )
    tasks_label = (
        f"TASK SUBSET {protocol['selected_tasks']}/{protocol['suite_task_count']}"
        if not protocol["full_task_suite"]
        else f"all {protocol['suite_task_count']} tasks"
    )
    rates = ", ".join(f"{hz:g}" for hz in protocol["observed_control_hz"]) or "not yet observed"
    pacing_label = f"{protocol['mode']} | control Hz: {rates} ({protocol['control_hz_source']})"
    title = (
        f"{protocol['suite_label']} | H={summary['metadata']['horizon']} | {limit_label} | {tasks_label}\n"
        f"{pacing_label}\n{coverage_label} | "
        f"Completed episodes: {coverage['completed_episodes']}/{coverage['expected_episodes']} | "
        + ", ".join(f"K={row['k']}: n={row['episodes']}" for row in rows)
    )
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    _prepare_axes(axes, rows)
    fig.suptitle(title, fontsize=15)
    success = _series(rows, "success_rate", "", 100)
    low = np.asarray(
        [np.nan if row["success_wilson95"][0] is None else 100 * row["success_wilson95"][0] for row in rows]
    )
    high = np.asarray(
        [np.nan if row["success_wilson95"][1] is None else 100 * row["success_wilson95"][1] for row in rows]
    )
    axes[0, 0].errorbar(
        x, success, yerr=[np.maximum(0, success - low), np.maximum(0, high - success)], fmt="o-", capsize=5
    )
    axes[0, 0].set(title="Success rate (episode Wilson 95% CI)", ylabel="Success (%)", ylim=(-4, 108))
    for position, row, rate in zip(x, rows, success, strict=True):
        axes[0, 0].annotate(
            f"{row['successes']}/{row['episodes']}" if row["episodes"] else "N/A",
            (position, rate if np.isfinite(rate) else 0),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )

    axes[0, 1].plot(
        x, _series(rows, "completion_wall_s_success_only"), "o-", label="Successful episodes only"
    )
    axes[0, 1].plot(x, _series(rows, "episode_wall_s_all"), "s--", label="All episodes (includes timeouts)")
    axes[0, 1].set(title="Mean wall time; successful counts shown", ylabel="Seconds")
    for position, row in zip(x, rows, strict=True):
        axes[0, 1].text(
            position,
            0.02,
            f"success n={row['successes']}",
            transform=axes[0, 1].get_xaxis_transform(),
            ha="center",
            fontsize=8,
        )
    axes[0, 1].legend(fontsize=9)

    for field, style in (("mean", "o-"), ("p95", "s--")):
        axes[0, 2].plot(x, _series(rows, "observation_age_ms", field), style, label=field)
    axes[0, 2].set(title="Observation age at action dispatch", ylabel="Milliseconds / emitted action")
    axes[0, 2].legend()

    for metric, label, color in (
        ("e2e_first_action_ms", "First dispatch E2E", "C0"),
        ("pipeline_ms", "Policy pipeline", "C1"),
    ):
        for field, style in (("mean", "o-"), ("p95", "s--")):
            axes[1, 0].plot(x, _series(rows, metric, field), style, color=color, label=f"{label} {field}")
    axes[1, 0].set(title="Per-plan latency", ylabel="Milliseconds / plan")
    axes[1, 0].legend(fontsize=8)

    for kind, style in (("boundary", "o-"), ("interior", "s--")):
        axes[1, 1].plot(x, _series(rows, f"{kind}_action_delta_l2"), style, label=f"{kind} mean")
        axes[1, 1].plot(x, _series(rows, f"{kind}_action_delta_l2", "p95"), ":", label=f"{kind} p95")
    axes[1, 1].set(title="Action discontinuity (arm 6D)", ylabel="L2 change in executed command")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].plot(
        x,
        _series(rows, "deadline_missed", "fraction", 100),
        "o-",
        color="C3",
        label="Actual dispatch (includes sleep)",
    )
    axes[1, 2].plot(
        x,
        _series(rows, "ready_deadline_missed", "fraction", 100),
        "s--",
        color="C4",
        label="Ready before pacing (diagnostic)",
    )
    axes[1, 2].set(title="Missed control deadlines", ylabel="Misses / emitted actions (%)", ylim=(-4, 104))
    axes[1, 2].legend(fontsize=8)
    scope_note = (
        "SMOKE success rates use a shortened episode limit; not standard LIBERO benchmark success rates.\n"
        if protocol["shortened_episode_limit"]
        else ""
    )
    fig.supxlabel(
        scope_note
        + "Gaps = no samples. Wall time includes inference, pacing and simulator stepping; reset/warmup excluded.\n"
        + "Simulation pauses during inference/pacing. Arm6 command differences exclude the gripper and are not physical jerk.\n"
        + (
            "Initial observation SHA-256 verified across available K: "
            f"{protocol['initial_observation_pairing_complete_groups']}/{protocol['initial_observation_pairing_groups']} groups have all requested K."
            if protocol["initial_observation_pairing_verified"]
            else "Initial observation pairing has not been verified by SHA-256."
        ),
        fontsize=10,
    )
    _save_figure(fig, output_dir, "comparison")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    _prepare_axes(axes, rows)
    fig.suptitle(
        f"{protocol['suite_label']} boundary diagnostics | {limit_label} | {tasks_label} | {coverage_label}\n{pacing_label}",
        fontsize=14,
    )
    for field, style in (("mean", "o-"), ("p95", "s--")):
        axes[0, 0].plot(x, _series(rows, "boundary_interval_abs_error_ms", field), style, label=field)
    axes[0, 0].set(
        title="Boundary dispatch interval jitter", ylabel="Absolute interval error vs control period (ms)"
    )
    axes[0, 0].legend()
    for axis, metric, label in (
        (axes[0, 1], "action_accel_l2", "Arm command second difference"),
        (axes[1, 0], "translation_delta_l2", "Translation command difference"),
        (axes[1, 1], "rotation_delta_l2", "Rotation command difference"),
    ):
        for kind, style in (("boundary", "o-"), ("interior", "s--")):
            axis.plot(x, _series(rows, f"{kind}_{metric}"), style, label=f"{kind} mean")
            axis.plot(x, _series(rows, f"{kind}_{metric}", "p95"), ":", label=f"{kind} p95")
        axis.set(title=label, ylabel="L2 in executed command units")
        axis.legend(fontsize=8)
    for kind, style in (("boundary", "o-"), ("interior", "s--")):
        axes[0, 2].plot(x, _series(rows, f"{kind}_gripper_changed", "fraction", 100), style, label=kind)
    axes[0, 2].set(
        title="Gripper command sign switches (separate)",
        ylabel="Transitions with sign switch (%)",
        ylim=(-4, 104),
    )
    axes[0, 2].legend()
    for field, style in (("mean", "o-"), ("p95", "s--")):
        axes[1, 2].plot(x, _series(rows, "dispatch_lateness_ms", field), style, label=field)
    axes[1, 2].set(title="Actual dispatch lateness, including sleep", ylabel="Milliseconds / emitted action")
    axes[1, 2].legend()
    fig.supxlabel(
        "First-ever action excluded. K=1 has no interior transitions. Command differences are not measured robot jerk.",
        fontsize=10,
    )
    _save_figure(fig, output_dir, "boundary_diagnostics")


def build_report(output_dir: Path) -> dict[str, Any]:
    """Read completed episodes and produce JSON/CSV summaries plus PNG/PDF figures."""
    output_dir = Path(output_dir)
    if (output_dir / "INVALIDATED.md").exists():
        raise ValueError(
            f"Refusing to regenerate an invalidated experiment report: {output_dir / 'INVALIDATED.md'}"
        )
    metadata = json.loads((output_dir / "metadata.json").read_text())
    suite = metadata.get("suite", "libero_10")
    suite_defaults = {"libero_10": (10, 520, "LIBERO Long"), "libero_90": (90, 400, "LIBERO 90")}
    if suite not in suite_defaults:
        raise ValueError(f"Unsupported LIBERO report suite: {suite}")
    default_task_count, default_max_steps, suite_label = suite_defaults[suite]
    suite_task_count = metadata.get("suite_task_count", default_task_count)
    suite_max_steps = metadata.get("suite_max_steps", default_max_steps)
    episodes: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    paired_hashes: dict[tuple[int, int], str] = {}
    paired_ks: dict[tuple[int, int], set[int]] = {}
    verify_pairing = metadata.get("schema_version", 1) >= 2
    for path in sorted((output_dir / "episodes").glob("task_*_episode_*_k_*.json")):
        episode = json.loads(path.read_text())
        identity = (episode["k"], episode["task_id"], episode["episode_index"])
        if identity in seen:
            raise ValueError(f"Duplicate episode identity {identity}: {path}")
        if episode["k"] not in metadata["ks"] or episode["task_id"] not in metadata["task_ids"]:
            raise ValueError(f"Episode outside metadata experiment configuration: {path}")
        if not 0 <= episode["episode_index"] < metadata["n_episodes"]:
            raise ValueError(f"Episode index outside metadata experiment configuration: {path}")
        if verify_pairing:
            initial_hash = episode.get("initial_observation_sha256")
            if (
                not isinstance(initial_hash, str)
                or len(initial_hash) != 64
                or any(character not in "0123456789abcdef" for character in initial_hash)
            ):
                raise ValueError(f"Missing or invalid initial observation SHA-256: {path}")
            pair = (episode["task_id"], episode["episode_index"])
            if pair in paired_hashes and paired_hashes[pair] != initial_hash:
                raise ValueError(
                    f"Initial observation SHA-256 mismatch across K for task/episode {pair}: {path}"
                )
            paired_hashes[pair] = initial_hash
            paired_ks.setdefault(pair, set()).add(episode["k"])
        seen.add(identity)
        episodes.append(episode)

    expected_per_k = len(metadata["task_ids"]) * metadata["n_episodes"]
    expected_total = expected_per_k * len(metadata["ks"])
    baseline = [episode for episode in episodes if episode["k"] == 40]
    by_k = []
    per_task = []
    for k in metadata["ks"]:
        selected = [episode for episode in episodes if episode["k"] == k]
        row = {"k": k, "expected_episodes": expected_per_k, **_aggregate(selected)}
        row["paired_vs_k40"] = _paired_wall_stats(selected, baseline)
        by_k.append(row)
        for task_id in metadata["task_ids"]:
            task_episodes = [episode for episode in selected if episode["task_id"] == task_id]
            per_task.append(
                {
                    "k": k,
                    "task_id": task_id,
                    "task_name": task_episodes[0]["task_name"] if task_episodes else None,
                    "expected_episodes": metadata["n_episodes"],
                    **_aggregate(task_episodes),
                }
            )

    counts = Counter(episode["k"] for episode in episodes)
    requested_hz = metadata.get("control_hz")
    summary = {
        "metadata": metadata,
        "protocol": {
            "suite": suite,
            "suite_label": suite_label,
            "suite_task_count": suite_task_count,
            "suite_max_steps": suite_max_steps,
            "selected_tasks": len(metadata["task_ids"]),
            "selected_task_ids": sorted(metadata["task_ids"]),
            "episodes_per_task_per_k": metadata["n_episodes"],
            "expected_rollouts_per_k": expected_per_k,
            "full_task_suite": set(metadata["task_ids"]) == set(range(suite_task_count)),
            "max_steps": metadata.get("max_steps", suite_max_steps),
            "shortened_episode_limit": metadata.get("max_steps", suite_max_steps) < suite_max_steps,
            "mode": metadata.get("mode", "unspecified"),
            "reset_protocol": metadata.get("reset_protocol", "unspecified"),
            "initial_observation_pairing_verified": verify_pairing and bool(episodes),
            "initial_observation_pairing_groups": len(paired_ks),
            "initial_observation_pairing_complete_groups": sum(
                group_ks == set(metadata["ks"]) for group_ks in paired_ks.values()
            ),
            "initial_observation_pairing_partial_groups": sum(
                group_ks != set(metadata["ks"]) for group_ks in paired_ks.values()
            ),
            "requested_control_hz": requested_hz,
            "observed_control_hz": sorted(
                {episode["control_hz"] for episode in episodes if episode.get("control_hz") is not None}
            ),
            "control_hz_source": "native LIBERO control_freq"
            if requested_hz is None
            else "CLI, verified against native control_freq",
        },
        "coverage": {
            "completed_episodes": len(episodes),
            "expected_episodes": expected_total,
            "complete": len(episodes) == expected_total,
            "completed_by_k": {str(k): counts[k] for k in metadata["ks"]},
            "missing_by_k": {str(k): expected_per_k - counts[k] for k in metadata["ks"]},
        },
        "definitions": {
            "success_rate": "Fraction of completed episodes. Incomplete runs can have unequal task coverage.",
            "benchmark_scope": "max_steps below the suite's standard episode limit is a shortened smoke experiment: its success rates are not standard suite benchmark rates. Standard limits are 520 for LIBERO Long and 400 for LIBERO 90. Task subsets remain subsets even when the requested run is complete.",
            "rollout_count": "n_episodes is per task per K. Expected rollouts equal selected task count times n_episodes times requested K count.",
            "initial_observation_pairing": "Schema >= 2 requires a SHA-256 initial observation hash in every episode and exact equality across all available K for each (task_id, episode_index). Verified means all available hashes passed; complete groups contain every requested K, partial groups are incomplete. The runner creates a fresh environment for each evaluated K/episode. INVALIDATED.md prevents report regeneration.",
            "pacing": "Synchronous wall-clock pacing: first action due one period after episode start; next action due one period after previous actual dispatch. Overruns rebase the schedule without catch-up bursts. Physics advances only in env.step and pauses during planning/sleep.",
            "control_hz": "Observed Hz comes from episode records. By default it is read from LIBERO's native control_freq; explicit CLI Hz must match native control_freq. Metadata control_hz=null means automatic selection, not missing observed Hz.",
            "success_wilson95": "Descriptive episode-level binomial CI; does not model within-task dependence.",
            "completion_wall_s_success_only": "Wall time from initial observation delivery to final successful env.step return; includes inference, pacing and simulator stepping, excludes reset and warmup. Selection depends on K.",
            "episode_wall_s_all": "All completed rollout durations, including timeouts; not a completion-time estimate.",
            "observation_age_ms": "Raw source observation delivery after reset/step to dispatch of each action that uses it; includes preprocessing and pacing. Action-weighted wall time, not camera exposure time.",
            "pipeline_ms": "Per-plan preprocessing, synchronized policy inference, postprocessing and CPU action transfer.",
            "e2e_first_action_ms": "Raw source observation delivery to first dispatched action from that plan, including pacing; plan-weighted.",
            "deadline_missed": "Actual dispatch exceeds scheduled deadline plus metadata.deadline_tolerance_ms, including sleep overshoot. Fraction denominator is emitted actions, not theoretical control slots.",
            "ready_deadline_missed": "Diagnostic: action-ready timestamp before pacing exceeds scheduled deadline plus tolerance; excludes subsequent sleep overshoot. Fraction denominator is emitted actions.",
            "boundary": "Transition into a new plan's first action, excluding the episode's first action.",
            "boundary_action_delta_l2": "Computed L2 adjacent executed arm6 command difference: three translation and three rotation components, gripper excluded. Includes intended motion; not measured physical jerk.",
            "boundary_action_accel_l2": "L2 second difference of arm commands; not physical robot acceleration or jerk.",
            "gripper_changed": "Separate binary gripper command sign-switch rate using (command > 0), for boundary and interior transitions.",
            "boundary_interval_abs_error_ms": "Absolute value of (inter-dispatch interval - target control period), sampled at chunk boundaries.",
            "paired_vs_k40": "Matches task, episode, seed and initial state; wall comparisons use pairs where both K and K=40 succeed.",
            "empty_samples": "null in JSON, empty CSV scalar cells and gaps in figures; never replaced with zero.",
        },
        "by_k": by_k,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    _write_csv(output_dir / "summary.csv", by_k)
    _write_csv(output_dir / "per_task_summary.csv", per_task)
    _write_csv(
        output_dir / "episodes.csv",
        [
            {key: value for key, value in episode.items() if key not in ("actions", "plans")}
            for episode in episodes
        ],
    )
    context_keys = ("k", "task_id", "task_name", "episode_index", "seed", "init_state_id", "success")
    for source in ("actions", "plans"):
        _write_csv(
            output_dir / f"{source}.csv",
            [
                {**{key: episode[key] for key in context_keys}, **row}
                for episode in episodes
                for row in episode[source]
            ],
        )
    _plot(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Experiment directory containing metadata.json and episodes/.",
    )
    args = parser.parse_args()
    summary = build_report(args.output_dir)
    coverage = summary["coverage"]
    print(
        f"Report: {args.output_dir} ({coverage['completed_episodes']}/{coverage['expected_episodes']} episodes)"
    )


if __name__ == "__main__":
    main()
