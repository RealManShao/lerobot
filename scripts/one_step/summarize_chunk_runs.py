"""Verify complete paired LIBERO-10 runs and export their metrics and per-episode evidence."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--episodes-per-task", type=int, required=True)
    args = parser.parse_args()
    resets = {}
    runs = []
    seed_states = {}
    for file in sorted(args.root.glob("seed*/*.json")):
        run = json.loads(file.read_text())
        if run.get("policy_type") != "demonstration_chunk":
            continue
        variant, seed = run["model"]["variant"], run["seed"]
        if run["action_offset"] != 1 or len(set(run["task_names"])) != 10:
            raise ValueError(f"Unexpected cache or task count in {file}")
        episodes = []
        calibration = {}
        outcome = defaultdict(dict)
        for rollout in sorted(file.parent.glob(f"{variant}-s*/episodes.json")):
            batch = json.loads(rollout.read_text())
            if (batch["variant"], batch["seed"], batch["H"], batch["K"]) != (variant, seed, 40, 5):
                raise ValueError(f"Protocol mismatch in {rollout}")
            calibration.update(batch["calibration"])
            for episode in batch["episodes"]:
                task, state, fingerprint = (episode[key] for key in ("task", "initial_state", "fingerprint"))
                if state in outcome[task] or resets.setdefault((task, state), fingerprint) != fingerprint:
                    raise ValueError(f"Duplicate or unpaired reset in {rollout}: {task}, {state}")
                outcome[task][state] = episode["success"]
                episodes.append(episode)
        if set(outcome) != set(run["task_names"]) or len({tuple(sorted(s)) for s in outcome.values()}) != 1:
            raise ValueError(f"Incomplete task-by-initial-state coverage in {file}")
        states = tuple(sorted(next(iter(outcome.values()))))
        if len(states) != args.episodes_per_task or states != seed_states.setdefault(seed, states):
            raise ValueError(f"Unexpected evaluation states in {file}")
        if any(c["sim_state_offset"] != 1 for c in calibration.values()):
            raise ValueError(f"Unexpected calibration timing in {file}")
        run.update(episodes=episodes, calibration=calibration, successes=sum(e["success"] for e in episodes))
        runs.append(run)
    if not runs:
        raise ValueError("No model reports found")
    args.out.write_text(json.dumps({"H": 40, "K": 5, "runs": runs}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
