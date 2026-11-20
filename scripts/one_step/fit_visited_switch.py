"""Fit sensor-only switch preferences from matched physical interventions."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def score(values, source="combined"):
    # Spatial shaping contributes at most 0.25; positive values indicate better outcomes.
    success, predicates, distance = values.T
    motion = -0.25 * distance / 0.4
    if source == "motion":
        return motion
    if source == "events":
        return success + predicates
    return success + predicates + motion


def load(root, initial_states):
    rows = []
    for task in range(10):
        for state in initial_states:
            rec = json.loads((root / f"collect-{task}-{state}.json").read_text())
            if rec["initial_state"] != state or len({r["tick"] for r in rec["rows"]}) != len(rec["rows"]):
                raise ValueError("Scenario or branch metadata mismatch")
            for r in rec["rows"]:
                rows.append((r["features"], r["direct"], r["servo"], state, task))
    return tuple(np.array(col) for col in zip(*rows, strict=True))


def solve(x, y, penalty):
    center, scale = x.mean(axis=0), x.std(axis=0).clip(1e-5)
    z = (x - center) / scale
    coef = np.linalg.solve(z.T @ z + len(x) * penalty * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    return center, scale, coef, y.mean()


def predict(parameters, x):
    center, scale, coef, bias = parameters
    return (x - center) / scale @ coef + bias


def evaluate(pred, y, a, b):
    pick = pred > 0
    return {
        "points": len(y),
        "pick_servo": int(pick.sum()),
        "servo_better_by_1e-4": int((y > 1e-4).sum()),
        "direct_better_by_1e-4": int((y < -1e-4).sum()),
        "sign_accuracy_beyond_1e-4": float(np.mean((pick == (y > 0))[abs(y) > 1e-4]))
        if (abs(y) > 1e-4).any()
        else None,
        "gain_over_direct": float(np.mean(pick * y)),
        "always_servo_gain": float(y.mean()),
        "hindsight_gain": float(y.clip(0).mean()),
        "predicate_diff_windows": int(np.count_nonzero(a[:, 1] != b[:, 1])),
        "completion_diff_windows": int(np.count_nonzero(a[:, 0] != b[:, 0])),
        "mean_absolute_distance_difference_mm": float(1000 * np.abs(a[:, 2] - b[:, 2]).mean()),
        "mean_absolute_utility_difference": float(abs(y).mean()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--head-directory", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.head_directory / "object_constant.json").read_text())
    head_hash = hashlib.sha256((args.head_directory / manifest["weights"]).read_bytes()).hexdigest()
    tr, test = load(args.root, range(16, 24)), load(args.root, range(24, 28))
    result = {
        "source": "20k object-coordinate policy",
        "fit_states": list(range(16, 24)),
        "held_states": list(range(24, 28)),
    }
    for name, h, source in (
        ("short", 0, "combined"),
        ("delayed", 1, "combined"),
        ("motion", 1, "motion"),
        ("events", 1, "events"),
    ):
        x, a, b, groups, _ = tr
        target = score(b[:, h], source) - score(a[:, h], source)
        cv = {}
        for lam in (0.01, 0.1, 1.0, 10.0):
            pred = np.zeros(len(x))
            for g in np.unique(groups):
                fit = groups != g
                pred[~fit] = predict(solve(x[fit], target[fit], lam), x[~fit])
            cv[lam] = np.mean((pred - target) ** 2)
        lam = min(cv, key=cv.get)
        pars = solve(x, target, lam)
        center, scale, coef, bias = pars
        x2, a2, b2, _, tasks = test
        held_pred = predict(pars, x2)
        held_target = score(b2[:, h], source) - score(a2[:, h], source)
        result[name] = {
            "mean": center.tolist(),
            "scale": scale.tolist(),
            "coef": coef.tolist(),
            "bias": float(bias),
            "lambda": lam,
            "cv_mse": cv,
            "fit_servo_fraction": float((predict(pars, x) > 0).mean()),
            "fit": evaluate(predict(pars, x), target, a[:, h], b[:, h]),
            "held": evaluate(held_pred, held_target, a2[:, h], b2[:, h]),
            "held_per_task": {
                int(t): evaluate(
                    held_pred[tasks == t], held_target[tasks == t], a2[tasks == t, h], b2[tasks == t, h]
                )
                for t in np.unique(tasks)
            },
            "fit_rows_per_state": dict(Counter(groups.tolist())),
        }
        gate = {k: result[name][k] for k in ("mean", "scale", "coef", "bias")}
        export = {"head_sha256": head_hash, "gate": gate}
        (args.root / f"{name}_switch.json").write_text(json.dumps(export, allow_nan=False, indent=2) + "\n")
    # Keep the fitted mean authority equal to the demonstration-calibrated constant.
    delayed = result["delayed"]
    rate = delayed["fit_servo_fraction"]
    base = manifest["global_weight"]
    span = min(base / rate, (1 - base) / (1 - rate)) if 0 < rate < 1 else 0.0
    low = max(0.0, base - rate * span)
    high = min(1.0, low + span)
    result["blend_weights"] = [low, high]
    cap_gate = {k: delayed[k] for k in ("mean", "scale", "coef", "bias")}
    cap_gate["weights"] = [low, high]
    (args.root / "bounded_switch.json").write_text(
        json.dumps({"head_sha256": head_hash, "gate": cap_gate}, indent=2) + "\n"
    )
    tmp = args.root / "gate.tmp"
    tmp.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
    tmp.replace(args.root / "gate.json")


if __name__ == "__main__":
    main()
