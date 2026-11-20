"""Positive control: execute recorded LIBERO commands from the first recorded simulator state."""

import argparse
import json
from pathlib import Path

import h5py
from benchmark_libero_feedback import _make_environment, _prepare_libero_resources
from cache_libero import TASKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-assets", required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--demos", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _prepare_libero_resources(args, args.out)
    from lerobot.envs.libero import _get_suite

    suite = _get_suite("libero_10")
    rows = []
    prefixes = (
        [p.stem.removesuffix("_demo") for p in sorted(args.suite.glob("*_demo.hdf5"))]
        if args.all_tasks
        else TASKS
    )
    for prefix in prefixes:
        (task,) = [j for j, item in enumerate(suite.tasks) if item.name.startswith(prefix)]
        (path,) = args.suite.glob(f"{prefix}*.hdf5")
        with h5py.File(path) as source:
            for number in range(args.demos):
                demo = source[f"data/demo_{number}"]
                env = _make_environment(suite, task, 520, 128)
                try:
                    env.reset(seed=number)
                    env._env.set_init_state(demo["states"][0])
                    steps = 0
                    for action in demo["actions"][:]:
                        steps += 1
                        _, _, done, _, info = env.step(action)
                        if done:
                            break
                    rows.append(
                        {"task": prefix, "demo": number, "steps": steps, "success": bool(info["is_success"])}
                    )
                finally:
                    env.close()
    (args.out / "expert.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
