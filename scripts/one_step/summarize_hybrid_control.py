"""Validate paired LIBERO rollouts, simulator branches and report actuator and routing contrasts."""

import argparse
import json
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from statistics import mean


def avg_dict(rows, fields):
    return {k: mean(r[k] for r in rows) for k in fields}


def paired(a, b):
    keys = sorted(a.keys() & b.keys())
    if not keys:
        raise ValueError("No common initial states")
    x = [a[k] for k in keys]
    y = [b[k] for k in keys]
    return {
        "n": len(keys),
        "successes": [sum(x), sum(y)],
        "a_only": sum(i and not j for i, j in zip(x, y, strict=True)),
        "b_only": sum(j and not i for i, j in zip(x, y, strict=True)),
    }


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    outcomes = defaultdict(dict)
    summary = {"seeds": {}}
    fingerprints = {}
    eval_count = 0
    for seed in (1, 42):
        folder = root / f"seed{seed}"
        runs = {}
        for path in sorted(folder.glob("*-s*/episodes.json")):
            label, first = path.parent.name.rsplit("-s", 1)
            first = int(first)
            r = read(path)
            expected_states = range(
                first,
                first
                + (
                    4
                    if first == 8
                    and (
                        label.startswith("outcome_")
                        or label in {"learned", "always_half", "history_constant", "history_adaptive"}
                    )
                    else 2
                ),
            )
            if r["seed"] != seed or r["K"] != 5 or r["H"] != 40 or len(r["calibration"]) != 10:
                raise ValueError(f"bad protocol {path}")
            if any(cal["sim_state_offset"] != 1 for cal in r["calibration"].values()):
                raise ValueError(f"bad timing {path}")
            expected = {(task, state) for task in r["calibration"] for state in expected_states}
            eps = r["episodes"]
            if len(eps) != len(expected) or {(e["task"], e["initial_state"]) for e in eps} != expected:
                raise ValueError(f"incomplete {path}")
            eval_count += len(eps)
            groups = defaultdict(list)
            transitions = []
            rates = []
            changes = []
            sequence_gains = []
            for e in eps:
                task, state = e["task"], e["initial_state"]
                key = (task, state)
                if fingerprints.setdefault(key, e["fingerprint"]) != e["fingerprint"]:
                    raise ValueError(f"Unpaired reset {path}, {key}")
                outcomes[label][seed, *key] = e["success"]
                if "gate_trace" in e:
                    trace = e["gate_trace"]
                    if len(trace) != (e["steps"] + 4) // 5:
                        raise ValueError(f"Missing replans {path}, {key}")
                    weights = [x[1] for x in trace]
                    rates.append(mean(weights))
                    changes.append(max(weights) - min(weights))
                    transitions.append(
                        sum(abs(a - b) >= 0.25 for a, b in pairwise(weights)) / max(1, len(weights) - 1)
                    )
                    initial = trace[0][0]
                    sequence_gains.append(any(mask & ~initial for mask, _, _ in trace) or e["success"])
                    for mask, w, _ in trace:
                        groups[task + "|" + str(mask)].append(w)
            out = {
                "episodes": len(eps),
                "success": sum(e["success"] for e in eps),
                "per_task": {
                    t: sum(e["success"] for e in eps if e["task"] == t) for t in sorted(r["calibration"])
                },
            }
            if rates:
                out.update(
                    mean_servo_weight=mean(rates),
                    episodes_with_weight_range_over_025=sum(v >= 0.25 for v in changes),
                    mean_switch_rate=mean(transitions),
                    episodes_gaining_goal_predicate=sum(sequence_gains),
                    weights_by_predicate={k: {"plans": len(v), "mean": mean(v)} for k, v in groups.items()},
                )
            runs.setdefault(label, {})[first] = out
        base = {"dual_prefix", "separate_prefix", "grip_event", "grip_binary"}
        swaps = {"dual_prefix-grip-separate_prefix", "separate_prefix-grip-dual_prefix"}
        outcomes_set = {"outcome_constant", "outcome_task", "outcome_phase", "outcome_context"}
        imitation = {"learned", "servo", "constant", "task", "adaptive", "hard"}
        history = {"history_servo", "history_half", "history_constant", "history_adaptive"}
        alignment = {"history_aligned", "history_matched"}
        all_names = base | swaps | outcomes_set | imitation | history | alignment
        if seed == 42:
            all_names.add("always_half")
        second = base | outcomes_set | {"learned", "history_constant", "history_adaptive"}
        if seed == 42:
            second.add("always_half")
        if set(runs) != all_names or any(
            set(runs[v]) != ({8} if v in alignment else {6, 8} if v in second else {6}) for v in runs
        ):
            raise ValueError(f"Unexpected run matrix for seed {seed}")
        offline = {}
        for file in sorted(folder.glob("*.json")):
            r = read(file)
            offline[file.stem] = {
                k: v
                for k, v in r.items()
                if k not in {"per_task", "task_names", "model", "policy_type", "seed"}
            }
            if "per_task" in r:
                offline[file.stem]["task_mean"] = avg_dict(
                    r["per_task"], ["arm_first5_mse", "arm_last20_mse", "switch_f1", "gripper_accuracy"]
                )
        summary["seeds"][seed] = {"rollouts": runs, "offline": offline}
    # Seed 1's outcome-optimal constant is exactly 0.5; reuse those episodes only for the matched comparison.
    if summary["seeds"][1]["offline"]["outcome_constant"]["global_choice"] != 1:
        raise ValueError("No fixed-half control for seed 1")
    outcomes["always_half"].update({k: v for k, v in outcomes["outcome_constant"].items() if k[0] == 1})
    comparisons = [
        ("separate_prefix", "dual_prefix"),
        ("grip_binary", "separate_prefix"),
        ("grip_event", "grip_binary"),
        ("grip_binary", "dual_prefix"),
        ("separate_prefix-grip-dual_prefix", "separate_prefix"),
        ("dual_prefix-grip-separate_prefix", "dual_prefix"),
        ("history_aligned", "history_matched"),
        ("history_aligned", "learned"),
        ("history_matched", "learned"),
        ("history_adaptive", "learned"),
        ("history_adaptive", "history_constant"),
        ("history_adaptive", "adaptive"),
        ("history_servo", "servo"),
        ("history_half", "always_half"),
        ("adaptive", "learned"),
        ("servo", "learned"),
        ("outcome_phase", "outcome_task"),
        ("outcome_phase", "learned"),
        ("outcome_phase", "always_half"),
        ("outcome_context", "outcome_task"),
        ("outcome_task", "outcome_constant"),
        ("outcome_context", "learned"),
        ("outcome_context", "always_half"),
        ("always_half", "learned"),
    ]
    summary["pairs"] = {}
    for a, b in comparisons:
        entry = {"all": paired(outcomes[a], outcomes[b])}
        for key, states in (("states_6_7", range(6, 8)), ("states_8_11", range(8, 12))):
            x = {k: v for k, v in outcomes[a].items() if k[-1] in states}
            y = {k: v for k, v in outcomes[b].items() if k[-1] in states}
            if x.keys() & y.keys():
                entry[key] = paired(x, y)
        summary["pairs"][a + " vs " + b] = entry
    all_rows = {split: [] for split in ("train", "test")}
    diag = {}
    for i in range(10):
        for split in all_rows:
            all_rows[split] += read(root / "interventions" / f"{i}-{split}.json")["rows"]
        train_ids = {r["demo"] for r in read(root / "interventions" / f"{i}-train.json")["rows"]}
        held = read(root / "interventions" / f"{i}-test.json")["rows"]
        if any(r["demo"] in train_ids for r in held):
            raise ValueError("Demonstration overlap in intervention set")
        oracle = read(root / "oracle" / f"{i}-test-oracle.json")["rows"]
        history = read(root / "history-interventions" / f"{i}-test.json")["rows"]
        for a, b, h in zip(held, oracle, history, strict=True):
            if any(a[k] != b[k] or a[k] != h[k] for k in ("demo", "index", "t", "bits")):
                raise ValueError("Mismatched oracle anchor")
            for seed in ("1", "42"):
                c, d = a["costs"][seed], b["costs"][seed]
                if c[0] != d[0] or c[0] != h["costs"][seed][0]:
                    raise ValueError("Independent replay of learned policy differs")
                for label, part in (
                    ("learned", c[0]),
                    ("half", c[1]),
                    ("predicted_goal", c[2]),
                    ("oracle_goal", d[1]),
                    ("history_goal", h["costs"][seed][2]),
                    ("history_half", h["costs"][seed][1]),
                ):
                    diag.setdefault(seed, defaultdict(list))[label].append(part)
    summary["interventions"] = {
        "counts": {s: len(arr) for s, arr in all_rows.items()},
        "predicate_counts": {s: dict(Counter(x["bits"] for x in arr)) for s, arr in all_rows.items()},
        "heldout_m2": {
            s: {k: [mean(pair[idx] for pair in v) for idx in range(2)] for k, v in modes.items()}
            for s, modes in diag.items()
        },
    }
    summary.update(evaluated_episodes=eval_count, distinct_resets=len(fingerprints))
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
