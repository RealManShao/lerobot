"""Verify matched branch interventions and report paired closed-loop task outcomes."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from fit_visited_switch import load, score


def read(path):
    return json.loads(path.read_text())


def paired(a, b, keys):
    return {
        "n": len(keys),
        "a": sum(a[k] for k in keys),
        "b": sum(b[k] for k in keys),
        "only_a": sum(a[k] and not b[k] for k in keys),
        "only_b": sum(b[k] and not a[k] for k in keys),
        "per_seed_net": {s: sum(int(a[k]) - int(b[k]) for k in keys if k[0] == s) for s in (1, 42)},
    }


def held_routing(root, gate):
    x, a, b, _, task = load(root, range(24, 28))
    values = {}
    dg = [score(b[:, h]) - score(a[:, h]) for h in range(2)]
    for name in ("short", "delayed", "motion", "events"):
        m = gate[name]
        select = (x - m["mean"]) / m["scale"] @ np.array(m["coef"]) + m["bias"] > 0
        for label, y in zip(("5", "30"), dg, strict=True):
            gain = np.mean(select * y)
            within = 0.0
            for t in np.unique(task):
                rows = task == t
                within += rows.mean() * np.mean((select[rows] - select[rows].mean()) * y[rows])
            unchanged = (a[:, int(label == "30"), :2] == b[:, int(label == "30"), :2]).all(1)
            no_event_cov = (
                np.mean((select[unchanged] - select[unchanged].mean()) * y[unchanged])
                if unchanged.any()
                else None
            )
            values[f"{name}_at_{label}"] = {
                "gain_over_direct": float(gain),
                "matched_random_gain_without_event_differences": float(no_event_cov)
                if no_event_cov is not None
                else None,
                "gain_over_matched_random": float(gain - select.mean() * y.mean()),
                "within_task_gain_over_matched_random": float(within),
                "servo_fraction": float(select.mean()),
                "hindsight_gain_over_direct": float(np.maximum(0, y).mean()),
                "always_servo_gain_over_direct": float(y.mean()),
                "completion_differences": int(
                    sum(a[:, int(label == "30"), 0] != b[:, int(label == "30"), 0])
                ),
                "predicate_differences": int(sum(a[:, int(label == "30"), 1] != b[:, int(label == "30"), 1])),
            }
    decisive = (np.abs(dg[0]) > 1e-4) & (np.abs(dg[1]) > 1e-4)
    values["both_decisive"] = int(decisive.sum())
    values["sign_reversals"] = int(sum((dg[0][decisive] > 0) != (dg[1][decisive] > 0)))
    values["rows"] = len(x)
    return values


def render(summary, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
    modes = {
        "direct": "Learned only",
        "servo": "Servo only",
        "constant": "Calibrated blend",
        "mean_delayed": "Rate-matched blend",
        "random_delayed": "Rate-matched random",
        "bounded": "Bounded switch",
        "random_bounded": "Bounded random",
        "short": "5-step switch",
        "delayed": "30-step switch",
        "motion": "Motion label",
        "events": "Task-event label",
    }
    for row, mode in enumerate(modes):
        by_seed = summary["closed_loop"][mode]["per_seed"]
        a, b = (100 * by_seed[s] / 80 for s in (1, 42))
        left.plot([a, b], [row, row], color="#abb4bd", linewidth=2)
        left.scatter(a, row, color="#20758a", marker="o", label="Seed 1" if row == 0 else None)
        left.scatter(b, row, color="#b45b33", marker="s", label="Seed 42" if row == 0 else None)
    left.set(
        yticks=range(len(modes)),
        yticklabels=list(modes.values()),
        xlabel="Success on 80 scenarios per seed (%)",
        title="Closed-loop LIBERO-10",
    )
    left.invert_yaxis()
    left.legend(frameon=False, fontsize=8)
    labels = ["5-step switch", "30-step switch", "Motion label", "Task-event label"]
    for j, seed in enumerate((1, 42)):
        vals = [
            1000 * summary["held"][seed][m + "_at_30"]["gain_over_matched_random"]
            for m in ("short", "delayed", "motion", "events")
        ]
        right.barh(
            np.arange(4) + (j - 0.5) * 0.33,
            vals,
            height=0.31,
            label=f"Seed {seed}",
            color=("#20758a", "#b45b33")[j],
        )
    right.set(
        yticks=range(4),
        yticklabels=labels,
        title="Held-out branch pairs",
        xlabel="30-step utility gain over rate-matched random (×1000)",
    )
    right.invert_yaxis()
    right.axvline(0, linewidth=0.7, color="#66717a")
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.savefig(destination, dpi=175)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--figure", type=Path)
    args = p.parse_args()
    modes = (
        "direct",
        "servo",
        "constant",
        "short",
        "delayed",
        "mean_delayed",
        "random_delayed",
        "bounded",
        "random_bounded",
        "motion",
        "events",
    )
    outcomes = {m: {} for m in modes}
    fractions = {m: [] for m in modes}
    summary = {"branches": {}, "held": {}, "closed_loop": {}, "paired": {}}
    reset_hash = {}
    split_fingerprints = {name: set() for name in ("fit", "held", "eval")}
    for seed in (1, 42):
        root = args.root / f"seed{seed}"
        gate = read(root / "gate.json")
        summary["held"][seed] = held_routing(root, gate)
        counts = Counter()
        for task in range(10):
            for state in range(16, 28):
                data = read(root / f"collect-{task}-{state}.json")
                if (data["seed"], data["initial_state"]) != (seed, state):
                    raise ValueError("Unmatched branch record")
                split = "fit" if state < 24 else "held"
                counts[split] += len(data["rows"])
                split_fingerprints[split].add((task, data["fingerprint"]))
                if any(len(row["features"]) != 20 or not row["fingerprint"] for row in data["rows"]):
                    raise ValueError("Incomplete intervention")
            sham = read(root / f"sham-{task}-24.json")
            for row in sham["rows"]:
                if row["direct"] != row["servo"]:
                    raise ValueError("Sham intervention altered the outcome")
            counts["sham"] += len(sham["rows"])
            for state in range(32, 40):
                key = (seed, task, state)
                for m in modes:
                    ep = read(root / f"{m}-{task}-{state}.json")
                    if (ep["seed"], ep["mode"], ep["initial_state"]) != (seed, m, state) or len(
                        ep["weights"]
                    ) != (ep["ticks"] + 4) // 5:
                        raise ValueError(f"Invalid rollout: {seed}/{m}/{task}/{state}")
                    if reset_hash.setdefault((task, state), ep["fingerprint"]) != ep["fingerprint"]:
                        raise ValueError(f"Different initial observation: {seed}/{m}/{task}/{state}")
                    split_fingerprints["eval"].add((task, ep["fingerprint"]))
                    outcomes[m][key] = bool(ep["success"])
                    fractions[m].append(np.mean(ep["weights"]))
        summary["branches"][seed] = counts
    for left, right in (("fit", "held"), ("fit", "eval"), ("held", "eval")):
        if split_fingerprints[left] & split_fingerprints[right]:
            raise ValueError(f"Identical initial observations across {left} and {right}")
    summary["reset_fingerprints_per_split"] = {k: len(v) for k, v in split_fingerprints.items()}
    keys = list(outcomes["direct"])
    for m, a in outcomes.items():
        summary["closed_loop"][m] = {
            "success": sum(a.values()),
            "total": len(a),
            "mean_weight_per_episode": float(np.mean(fractions[m])),
            "states32_35": sum(v for k, v in a.items() if k[2] < 36),
            "states36_39": sum(v for k, v in a.items() if k[2] >= 36),
            "per_seed": {s: sum(a[k] for k in keys if k[0] == s) for s in (1, 42)},
            "per_task": [sum(a[k] for k in keys if k[1] == t) for t in range(10)],
        }
    for a, b in [
        ("short", "direct"),
        ("motion", "delayed"),
        ("events", "delayed"),
        ("delayed", "direct"),
        ("delayed", "short"),
        ("delayed", "constant"),
        ("delayed", "random_delayed"),
        ("delayed", "mean_delayed"),
        ("bounded", "constant"),
        ("bounded", "random_bounded"),
        ("bounded", "delayed"),
    ]:
        summary["paired"][f"{a} vs {b}"] = paired(outcomes[a], outcomes[b], keys)
    summary["episodes"] = sum(map(len, outcomes.values()))
    if args.figure:
        render(summary, args.figure)
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
