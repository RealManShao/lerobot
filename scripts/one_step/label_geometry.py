"""Reconstruct task-entity frames and gripper/owner contact at cached RGB timestamps."""

import argparse
from pathlib import Path

import h5py
import numpy as np


def entities(domain):
    names = list(
        dict.fromkeys(name for predicate in domain.parsed_problem["goal_state"] for name in predicate[1:])
    )
    model = domain.sim.model
    rows = []
    for name in names:
        state = domain.object_states_dict[name]
        if hasattr(state, "parent_name"):
            kind, index = "site", model.site_name2id(name)
            owner = model.site_bodyid[index]
            # Contact with the articulated link owning the region, including its descendants.
        else:
            kind, index = "body", domain.obj_body_id[name]
            owner = index
        family = {owner}
        for body in range(1, model.nbody):
            parent = body
            while parent > 0 and parent != owner:
                parent = model.body_parentid[parent]
            if owner != 0 and parent == owner:
                family.add(body)
        geoms = {i for i, b in enumerate(model.geom_bodyid) if b in family}
        rows.append((kind, index, geoms))
    fingers = {
        model.geom_name2id(n)
        for n in model.geom_names
        if n and n.startswith("gripper") and "collision" in n and ("finger" in n or "pad" in n)
    }
    if not fingers or len(names) > 4:
        raise ValueError((names, fingers))
    return names, rows, fingers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("suite", "assets", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--task", type=int, required=True)
    a = p.parse_args()
    import libero.libero as libero
    import robosuite
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs.env_wrapper import ControlEnv
    from probe import relocate

    libero._assets_path_cache = str(a.assets)
    path = sorted(a.suite.glob("*_demo.hdf5"))[a.task]
    suite = get_benchmark_dict()["libero_10"]()
    (task,) = [t for t in suite.tasks if t.name + "_demo" == path.stem]
    bddl = Path(libero.get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = ControlEnv(
        bddl_file_name=str(bddl),
        use_camera_obs=False,
        has_offscreen_renderer=False,
        camera_names=[],
        ignore_done=True,
    )
    a.out.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(path) as f:
            group = f["data"]
            names = sorted(group, key=lambda n: int(n.split("_")[-1]))
            order = np.random.default_rng(2026).permutation(names).tolist()
            for split, demos in (("train", order[:40]), ("test", order[40:])):
                poses = []
                touches = []
                ends = []
                ids = []
                errors = []
                start_ids = []
                for name in demos:
                    demo = group[name]
                    env.reset()
                    env.reset_from_xml_string(
                        relocate(
                            demo.attrs["model_file"],
                            a.assets,
                            Path(robosuite.__file__).parent / "models/assets",
                        )
                    )
                    labels, slots, fingers = entities(env.env)
                    xyz = demo["obs/ee_pos"][:]
                    starts = np.arange(0, len(xyz) - 40, 8)
                    states = demo["states"][:]
                    for t in starts:
                        env.sim.set_state_from_flattened(states[t + 1])
                        env.sim.forward()
                        data = env.sim.data
                        errors.append(np.linalg.norm(data.site_xpos[env.robots[0].eef_site_id] - xyz[t]))
                        other = set()
                        for c in data.contact[: data.ncon]:
                            if c.geom1 in fingers:
                                other.add(c.geom2)
                            if c.geom2 in fingers:
                                other.add(c.geom1)
                        pose = np.zeros((4, 9), dtype=np.float32)
                        touch = np.zeros(4, dtype=np.float32)
                        for j, (kind, k, geoms) in enumerate(slots):
                            pos, rot = (
                                (data.site_xpos[k], data.site_xmat[k])
                                if kind == "site"
                                else (data.body_xpos[k], data.body_xmat[k])
                            )
                            pose[j] = np.r_[pos, rot.reshape(3, 3)[:, :2].T.flatten()]
                            touch[j] = bool(geoms & other)
                        poses.append(pose)
                        touches.append(touch)
                        ends.append(xyz[t + 5])
                        ids.append(int(name.split("_")[-1]))
                        start_ids.append(t)
                dest = a.out / f"{a.task}-{split}.npz"
                np.savez_compressed(
                    dest,
                    label_version=2,
                    pose=np.stack(poses),
                    touch=np.stack(touches),
                    goal=np.stack(ends).astype("float32"),
                    episode=np.array(ids),
                    start=np.array(start_ids),
                    names=np.array(labels),
                    alignment=np.array(errors),
                    task=path.stem,
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
