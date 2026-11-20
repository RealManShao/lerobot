"""Collect or evaluate on-policy interventions. Simulator supervision lives in this experiment only."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from benchmark_libero_feedback import _make_environment, _prepare_libero_resources, observation_fingerprint
from eval_libero import robot_state
from label_geometry import entities

from lerobot.policies.demonstration_chunk.observed_control import ObservedController


def command(controller, obs):
    return controller.act(obs["pixels"]["image"], obs["pixels"]["image2"], robot_state(obs))


def initialize(suite, task, state):
    seed = 9000 + state
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = _make_environment(suite, task, 520, 128)
    env.init_state_id = state
    obs, _ = env.reset(seed=seed)
    return env, obs


class Progress:
    def __init__(self, env):
        self.domain = env._env.env
        self.predicates = self.domain.parsed_problem["goal_state"]
        names, slots, _ = entities(self.domain)
        self.slots = dict(zip(names, slots, strict=True))

    def position(self, name):
        kind, index, _ = self.slots[name]
        data = self.domain.sim.data
        return (data.site_xpos if kind == "site" else data.body_xpos)[index]

    def measure(self):
        bits = [bool(self.domain._eval_predicate(p)) for p in self.predicates]
        distances = [
            np.linalg.norm(self.position(p[1]) - self.position(p[2]))
            for p in self.predicates
            if p[0].lower() in ("in", "on")
        ]
        return [float(all(bits)), float(np.mean(bits)), float(np.mean(np.minimum(distances, 0.4)))]


def physics(env):
    return env._env.env.sim.get_state().flatten().copy()


def evaluate(suite, task, state, policy, random_switch=False):
    env, obs = initialize(suite, task, state)
    try:
        fp = observation_fingerprint(obs)
        policy.reset()
        weights = []
        rng = np.random.default_rng(5000 + state)
        probability = policy.weight
        for t in range(520):
            if random_switch and t % 5 == 0:
                policy.weight = float(rng.random() < probability)
            action = command(policy, obs)
            if t % 5 == 0:
                weights.append(policy.last_weight)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        policy.weight = probability
        return {"fingerprint": fp, "success": bool(info["is_success"]), "ticks": t + 1, "weights": weights}
    finally:
        env.close()


def collect(suite, task, state, policy, sham=False):
    # Four temporal strata. Their jitter uses scenario metadata only for experimental sampling.
    rng = np.random.default_rng(127 * task + state)
    times = [5 * int(rng.integers(a, b)) for a, b in [(2, 9), (11, 24), (26, 46), (48, 70)]]
    if sham:
        times = times[1:2]
    env, obs = initialize(suite, task, state)
    try:
        start = observation_fingerprint(obs)
        progress = Progress(env)
        actions, marks, anchors = [], [progress.measure()], {}
        factual = {}
        policy.reset()
        policy.weight = 0.0
        for t in range(times[-1] + 30):
            action = command(policy, obs)
            if t in times:
                anchors[t] = (physics(env), observation_fingerprint(obs), policy.features.tolist())
            actions.append(action.copy())
            obs, _, done, trunc, _ = env.step(action)
            marks.append(progress.measure())
            if sham and (t + 1 in (times[0] + 5, times[0] + 30) or done or trunc):
                factual[t + 1] = (physics(env), observation_fingerprint(obs))
            if done or trunc:
                break
    finally:
        env.close()
    rows = []
    for t, (q, fp, features) in anchors.items():
        # Replay exactly the factual controls from the same official reset. Check physics and both RGB views.
        replay = None
        for _ in range(3):
            replay, obs = initialize(suite, task, state)
            if observation_fingerprint(obs) == start:
                for act in actions[:t]:
                    obs, _, _, _, _ = replay.step(act.copy())
                if np.array_equal(physics(replay), q) and observation_fingerprint(obs) == fp:
                    break
            replay.close()
        else:
            raise RuntimeError(f"Replay mismatch at task {task} initial state {state} tick {t}")
        try:
            probe = Progress(replay)
            alternative = [probe.measure()]
            np.testing.assert_allclose(alternative[0], marks[t], rtol=0, atol=1e-10)
            policy.reset()
            for k in range(30):
                policy.weight = float(k < 5 and not sham)
                act = command(policy, obs)
                if k == 0:
                    np.testing.assert_allclose(policy.features, features, atol=1e-5, rtol=1e-5)
                obs, _, done, trunc, _ = replay.step(act)
                alternative.append(probe.measure())
                if sham and t + k + 1 in factual:
                    q_ref, fp_ref = factual[t + k + 1]
                    if not np.array_equal(physics(replay), q_ref) or observation_fingerprint(obs) != fp_ref:
                        raise RuntimeError("Unchanged control diverged in the sham replay")
                if done or trunc:
                    break
            rows.append(
                {
                    "tick": t,
                    "features": features,
                    "fingerprint": fp,
                    "start": marks[t],
                    "direct": [marks[min(t + h, len(marks) - 1)] for h in (5, 30)],
                    "servo": [alternative[min(h, len(alternative) - 1)] for h in (5, 30)],
                }
            )
        finally:
            replay.close()
    return {"fingerprint": start, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "checkpoint", "out"):
        p.add_argument("--" + field, required=True, type=Path)
    p.add_argument("--libero-assets", required=True)
    p.add_argument("--tasks", type=int, nargs="+", required=True)
    p.add_argument("--states", type=int, nargs="+", required=True)
    p.add_argument(
        "--mode",
        required=True,
        choices=[
            "collect",
            "sham",
            "direct",
            "servo",
            "constant",
            "short",
            "delayed",
            "mean_delayed",
            "random_delayed",
        ],
    )
    p.add_argument("--gate", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    resources = args.out / "resources" / f"{args.mode}-{args.tasks[0]}-{args.states[0]}"
    resources.mkdir(parents=True, exist_ok=True)
    _prepare_libero_resources(args, resources)
    from lerobot.envs.libero import _get_suite

    suite = _get_suite("libero_10")
    manifest = json.loads((args.root / "object_constant.json").read_text())
    switch_file = args.out / f"{args.mode}_switch.json" if args.mode in ("short", "delayed") else None
    policy = ObservedController.from_checkpoints(args.checkpoint, args.root, switch_file)
    if args.mode in ("direct", "collect", "sham", "servo"):
        policy.weight = float(args.mode == "servo")
    if args.mode in ("mean_delayed", "random_delayed"):
        research = json.loads(args.gate.read_text())
        policy.weight = research["delayed"]["fit_servo_fraction"]
    for i in args.tasks:
        name = manifest["task_names"][i]
        (task,) = [j for j, item in enumerate(suite.tasks) if item.name == name]
        for state in args.states:
            dst = args.out / f"{args.mode}-{i}-{state}.json"
            if dst.exists():
                continue
            value = (
                collect(suite, task, state, policy, sham=args.mode == "sham")
                if args.mode in ("collect", "sham")
                else evaluate(suite, task, state, policy, random_switch=args.mode == "random_delayed")
            )
            value.update(task=name, seed=manifest["seed"], initial_state=state, mode=args.mode)
            tmp = dst.with_suffix(".tmp")
            tmp.write_text(json.dumps(value, allow_nan=False) + "\n")
            tmp.replace(dst)


if __name__ == "__main__":
    main()
