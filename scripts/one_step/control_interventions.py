"""Evaluate model/learned translation mixtures on identical headless MuJoCo branches."""

import argparse
import json
import os
import traceback
from pathlib import Path

import h5py
import numpy as np
from probe import relocate

from lerobot.policies.demonstration_chunk.waypoint import waypoint_translation


def branch(env, command, object_ids, goal=None, inverse=None, weight=0):
    # This process and its children have no GPU context. Fork preserves OSC and solver state.
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read)
        try:
            eef = env.robots[0].eef_site_id
            for k, nominal in enumerate(command):
                action = np.array(nominal, dtype=float)
                if weight:
                    servo = waypoint_translation(env.sim.data.site_xpos[eef], goal, inverse, 5 - k)
                    action[:3] += weight * (servo - action[:3])
                env.step(action)
            pos = env.sim.data.body_xpos[object_ids].copy()
            rot = env.sim.data.body_xmat[object_ids].reshape(-1, 3, 3)
            markers = np.concatenate((pos[:, None], pos[:, None] + 0.05 * rot.transpose(0, 2, 1)), 1)
            value = {"eef": env.sim.data.site_xpos[eef].tolist(), "objects": markers.tolist()}
        except BaseException:
            value = {"error": traceback.format_exc()}
        with os.fdopen(write, "w") as f:
            json.dump(value, f, allow_nan=False)
        os._exit(0)
    os.close(write)
    with os.fdopen(read) as f:
        payload = f.read()
    _, status = os.waitpid(pid, 0)
    if status or not payload:
        raise RuntimeError(f"Branch process failed with status {status}")
    result = json.loads(payload)
    if "error" in result:
        raise RuntimeError(result["error"])
    return {k: np.asarray(v) for k, v in result.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("suite", "assets", "proposals", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--task-ids", nargs="+", type=int, required=True)
    p.add_argument(
        "--oracle-waypoint",
        action="store_true",
        help="Held-out diagnostic: supply the endpoint of an expert simulator branch",
    )
    args = p.parse_args()
    import libero.libero as libero
    import robosuite
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs.env_wrapper import ControlEnv

    libero._assets_path_cache = str(args.assets)
    suite = get_benchmark_dict()["libero_10"]()
    args.out.mkdir(parents=True, exist_ok=True)
    for i in args.task_ids:
        for split in ("test",) if args.oracle_waypoint else ("train", "test"):
            src = json.loads((args.proposals / f"{i}-{split}.json").read_text())
            suffix = "-oracle" if args.oracle_waypoint else ""
            dest = args.out / f"{i}-{split}{suffix}.json"
            if dest.exists():
                continue
            (task,) = [t for t in suite.tasks if t.name + "_demo" == src["task"]]
            bddl = Path(libero.get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = ControlEnv(
                bddl_file_name=str(bddl),
                use_camera_obs=False,
                has_offscreen_renderer=False,
                camera_names=[],
                ignore_done=True,
            )
            rows = []
            try:
                with h5py.File(args.suite / (src["task"] + ".hdf5")) as f:
                    previous = None
                    for row in src["rows"]:
                        demo = f["data"][row["demo"]]
                        if row["demo"] != previous:
                            env.reset()
                            xml = relocate(
                                demo.attrs["model_file"],
                                args.assets,
                                Path(robosuite.__file__).parent / "models/assets",
                            )
                            env.reset_from_xml_string(xml)
                            model = env.sim.model
                            ids = [
                                model.body_name2id(n)
                                for n in model.body_names
                                if model.body_jntnum[model.body_name2id(n)] > 0
                                and not n.startswith(("robot", "gripper", "mount"))
                            ]
                            if not ids:
                                raise ValueError("No movable bodies for consequence metric")
                            previous = row["demo"]
                        t = row["t"] + 1
                        for j in range(max(0, t - 3), t):
                            env.set_init_state(demo["states"][j])
                            env.step(demo["actions"][j])
                        obs = env.set_init_state(demo["states"][t])
                        align = np.linalg.norm(obs["robot0_eef_pos"] - demo["obs/ee_pos"][t - 1])
                        predicates = env.env.parsed_problem["goal_state"]
                        bits = sum(
                            int(bool(env.env._eval_predicate(g))) << j for j, g in enumerate(predicates)
                        )
                        expert = branch(env, demo["actions"][t : t + 5], ids)
                        out = {
                            "index": row["index"],
                            "demo": row["demo"],
                            "t": t - 1,
                            "bits": bits,
                            "alignment_m": float(align),
                            "costs": {},
                        }
                        for seed, pred in row["experts"].items():
                            costs = []
                            goal = expert["eef"] if args.oracle_waypoint else np.asarray(pred["goal"])
                            for w in (0.0, 1.0) if args.oracle_waypoint else (0.0, 0.5, 1.0):
                                actual = branch(
                                    env,
                                    pred["command"],
                                    ids,
                                    goal,
                                    np.asarray(src["inverse"]),
                                    w,
                                )
                                costs.append(
                                    [
                                        float(np.square(actual["eef"] - expert["eef"]).sum()),
                                        float(
                                            np.square(actual["objects"] - expert["objects"]).sum(-1).mean()
                                        ),
                                    ]
                                )
                            out["costs"][seed] = costs
                        rows.append(out)
            finally:
                env.close()
            tmp = dest.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"task": task.name, "predicates": predicates, "rows": rows}, allow_nan=False)
                + "\n"
            )
            tmp.replace(dest)


if __name__ == "__main__":
    main()
