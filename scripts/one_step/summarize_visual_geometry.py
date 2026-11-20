"""Verify crossed task-blind rollouts and summarize the long-training curves."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text())


def compare(a, b):
    if a.keys() != b.keys():
        raise ValueError("Unpaired evaluation sets")
    task_diffs = defaultdict(int)
    seed_diffs = defaultdict(int)
    for k in a:
        seed, task, _ = k
        task_diffs[task] += a[k] - b[k]
        seed_diffs[seed] += a[k] - b[k]
    return {
        "task_net": dict(task_diffs),
        "seed_net": dict(seed_diffs),
        "n": len(a),
        "a": sum(a.values()),
        "b": sum(b.values()),
        "a_only": sum(a[k] and not b[k] for k in a),
        "b_only": sum(b[k] and not a[k] for k in a),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    summary = {"curves": {}, "pairs": {}, "diagnostics": {}, "decomposition": {}}
    outcomes = defaultdict(dict)
    fingerprints = {}
    for g in ("world", "object", "consistent", "shared"):
        curves = {}
        for step in (1800, 7200, 20000):
            routes = {}
            modes = ["direct", "half", "constant", "contact"] + (["entity_contact"] if step == 20000 else [])
            if g == "shared":
                modes = ["direct"]
            for mode in modes:
                total = 0
                success = 0
                fields = defaultdict(float)
                tasks = defaultdict(int)
                chosen = []
                spread = []
                by_seed = {}
                weights = {}
                for seed in (1, 42):
                    folder = root / f"step{step}" / f"seed{seed}"
                    report = read(folder / f"{g}_{mode}.json")
                    if (report["steps"], report["seed"], report["model"]) != (
                        step,
                        seed,
                        {"geometry": g, "route": mode},
                    ):
                        raise ValueError("Manifest mismatch")
                    weights[seed] = report.get(
                        "entity_weight",
                        report["stage_weights"]
                        if mode == "contact"
                        else report["global_weight"]
                        if mode == "constant"
                        else 0
                        if mode == "direct"
                        else 0.5,
                    )
                    by_seed[seed] = {}
                    for first in (
                        [12, 14]
                        if step == 20000
                        or mode == "contact"
                        or (g in {"world", "shared"} and mode == "direct")
                        else [12]
                    ):
                        run = read(folder / f"{g}_{mode}-s{first}" / "episodes.json")
                        if run["task_id_input"] is not False or (
                            run["training_steps"],
                            run["H"],
                            run["K"],
                        ) != (step, 40, 5):
                            raise ValueError("Unexpected inference protocol")
                        eps = run["episodes"]
                        expected = {(t, n) for t in report["task_names"] for n in (first, first + 1)}
                        if len(eps) != 20 or {(e["task"], e["initial_state"]) for e in eps} != expected:
                            raise ValueError(f"Incomplete run: {folder}, {g}, {mode}, {first}")
                        by_seed[seed][first] = sum(e["success"] for e in eps)
                        for ep in eps:
                            task, state = ep["task"], ep["initial_state"]
                            if fingerprints.setdefault((task, state), ep["fingerprint"]) != ep["fingerprint"]:
                                raise ValueError("Reset fingerprint differs")
                            outcomes[g, step, mode][seed, task, state] = int(ep["success"])
                            success += ep["success"]
                            total += 1
                            tasks[task] += ep["success"]
                            stats = ep["perception_totals"]
                            trace = ep["gate_trace"]
                            if len(trace) != stats["plans"] or len(trace) != (ep["steps"] + 4) // 5:
                                raise ValueError("Incomplete control or perception trace")
                            for k, x in stats.items():
                                fields[k] += x
                            alpha = [x[1] for x in trace]
                            chosen.append(mean(alpha))
                            spread.append(max(alpha) - min(alpha))
                f1 = 2 * fields["true_positive"] / max(1, fields["contact"] + fields["called_contact"])
                routes[mode] = {
                    "success": success,
                    "episodes": total,
                    "by_seed_and_first_state": by_seed,
                    "per_task": dict(tasks),
                    "initial_12_13_success": sum(x[12] for x in by_seed.values()),
                    "gate_values": weights,
                    "average_rollout_weight": mean(chosen),
                    "episodes_weight_range_gt_025": sum(x > 0.25 for x in spread),
                    "live_pose_mm": fields["position_m"] * 1000 / fields["slots"],
                    "live_rotation_deg": fields["angle_deg"] / fields["slots"],
                    "live_contact_f1": f1,
                    "live_contact_prevalence": fields["contact"] / fields["plans"],
                    "live_plans": int(fields["plans"]),
                }
            curves[step] = routes
        summary["curves"][g] = curves
        summary["diagnostics"][g] = {s: read(root / f"diagnostic-{g}-{s}.json") for s in (1, 42)}
    for g in ("world", "object", "consistent"):
        for mode in ("contact", "constant", "half", "direct"):
            ref = outcomes[g, 1800, mode]
            later = {k: v for k, v in outcomes[g, 20000, mode].items() if k in ref}
            summary["pairs"][f"{g}/{mode}: 20k vs 1.8k on common states"] = compare(later, ref)
        for mode in ("entity_contact", "contact", "half"):
            summary["pairs"][f"{g}/{mode} vs calibrated constant 20k"] = compare(
                outcomes[g, 20000, mode], outcomes[g, 20000, "constant"]
            )
    for step in (1800, 7200, 20000):
        summary["pairs"][f"separate vs shared direct {step}"] = compare(
            outcomes["world", step, "direct"], outcomes["shared", step, "direct"]
        )
    for mode in ("direct", "half", "constant", "contact", "entity_contact"):
        summary["pairs"][f"consistent vs object 20k {mode}"] = compare(
            outcomes["consistent", 20000, mode], outcomes["object", 20000, mode]
        )
        summary["pairs"][f"object vs world 20k {mode}"] = compare(
            outcomes["object", 20000, mode], outcomes["world", 20000, mode]
        )
    for g in ("world", "object", "consistent"):
        summary["pairs"][f"{g}/calibrated constant vs direct 20k"] = compare(
            outcomes[g, 20000, "constant"], outcomes[g, 20000, "direct"]
        )
    summary["decomposition"] = {
        g: read(root / file)
        for g, file in [("object", "decomposition.json"), ("consistent", "decomposition-consistent.json")]
    }
    summary["gate_projection"] = read(root / "gate_projection.json")
    summary["rollouts"] = sum(len(x) for x in outcomes.values())
    summary["distinct_task_states"] = len(fingerprints)
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
