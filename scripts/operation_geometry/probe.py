"""Finite differences of LIBERO task consequences in a 12D action subspace."""

import argparse
import json
import os
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

H = 16
DIM = 12
RADII = (0.01, 0.04, 0.16)
FIT = 0.02


def anchors(actions):
    switches = np.flatnonzero(np.diff(actions[:, 6] > 0)) + 1
    valid = switches[(switches >= 24) & (switches + 52 <= len(actions))]
    if not len(valid):
        raise ValueError("No complete pre/post-switch windows")
    closing = valid[actions[valid, 6] > 0]
    t = int(closing[0] if len(closing) else valid[0])
    return (t - 24, t - 4, t + 12)


def relocate(xml, assets, robosuite_assets):
    root = ET.fromstring(xml)
    for elem in root.iter():
        file = elem.get("file")
        if file and not Path(file).is_file():
            folder = robosuite_assets if "robosuite" in file else assets
            path = folder / file.split("/assets/")[-1]
            if not path.is_file():
                raise FileNotFoundError(path)
            elem.set("file", str(path.resolve()))
    return ET.tostring(root, encoding="unicode")


def consequences(env, ids):
    d = env.sim.data
    p = d.body_xpos[ids].copy()
    rot = d.body_xmat[ids].reshape(-1, 3, 3)
    # Mean-squared world-space distances of four rigid markers per moving body.
    markers = np.concatenate((p[:, None, :], p[:, None, :] + 0.05 * rot.transpose(0, 2, 1)), 1)
    return [d.site_xpos[env.robots[0].eef_site_id].copy(), markers.flatten() / np.sqrt(4 * len(ids))]


def contact_signature(env, robot_geoms):
    data = env.sim.data
    pairs = set()
    for c in data.contact[: data.ncon]:
        a, b = int(c.geom1), int(c.geom2)
        if (a in robot_geoms) != (b in robot_geoms):
            pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


def rollout(env, commands, ids, robot_geoms):
    # Linux fork of a CPU-only simulator gives each probe identical MuJoCo, OSC,
    # gripper, fixture and solver state, including state absent from HDF5.
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            outcomes, signatures = [], []
            for i, command in enumerate(commands, 1):
                env.step(command)
                signatures.append(contact_signature(env, robot_geoms))
                if i in (H // 2, H):
                    outcomes.append([x.tolist() for x in consequences(env, ids)])
            result = {"outcomes": outcomes, "contacts": signatures}
        except BaseException:
            result = {"error": traceback.format_exc()}
        with os.fdopen(write_fd, "w") as stream:
            json.dump(result, stream, allow_nan=False)
        os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd) as stream:
        payload = stream.read()
    _, status = os.waitpid(pid, 0)
    if status or not payload:
        raise RuntimeError(f"Simulator clone failed with status {status}")
    r = json.loads(payload)
    if "error" in r:
        raise RuntimeError(r["error"])
    y = np.concatenate([np.r_[eef, objects] for eef, objects in r["outcomes"]]) / np.sqrt(2)
    return y, r["contacts"]


def probe(env, demo, t, rng, previous=None, scale_check=False):
    # Initialize the controller/gripper with real preceding commands while resetting
    # the recorded physical state at each step. The final state is the anchor.
    actions = demo["actions"]
    for i in range(max(0, t - 3), t):
        env.set_init_state(demo["states"][i])
        env.step(actions[i])
    obs = env.set_init_state(demo["states"][t])
    pose_error = np.linalg.norm(obs["robot0_eef_pos"] - demo["obs/ee_pos"][t])
    model = env.sim.model
    names = [
        n
        for n in model.body_names
        if model.body_jntnum[model.body_name2id(n)] > 0 and not n.startswith(("robot", "gripper", "mount"))
    ]
    if not names:
        raise ValueError("No moving non-robot bodies")
    ids = [model.body_name2id(n) for n in names]
    robot_geoms = {
        model.geom_name2id(n) for n in model.geom_names if n and n.startswith(("robot", "gripper"))
    }
    base = np.asarray(actions[t : t + H], dtype=float)
    # Contract perturbations only, so both signs stay in the actuator command box.
    slack = np.clip((1 - np.abs(base[:, :3])) / max(RADII), 0, 1)
    basis = np.eye(DIM).reshape(DIM, 4, 3).repeat(4, axis=1) * slack
    y0, c0 = rollout(env, base, ids, robot_geoms)

    def central(v, eps):
        pert = np.zeros_like(base)
        pert[:, :3] = eps * np.einsum("i,ith->th", v, basis)
        plus, cp = rollout(env, base + pert, ids, robot_geoms)
        minus, cm = rollout(env, base - pert, ids, robot_geoms)
        return {
            "v": v.tolist(),
            "eps": eps,
            "plus": (plus - y0).tolist(),
            "minus": (minus - y0).tolist(),
            "contact_changed": cp != c0 or cm != c0,
        }

    if previous is not None:
        if names != previous["body_names"] or not np.array_equal(basis, previous["basis"]):
            raise ValueError("Different action or consequence coordinates on second pass")
        if scale_check:
            check = central(np.eye(DIM)[0], FIT)
            old = previous["fit"][0]
            err = max(np.max(np.abs(np.asarray(check[k]) - old[k])) for k in ("plus", "minus"))
            if err > 1e-9:
                raise ValueError(f"Calibration across processes differs by {err}")
            return {
                "cross_run_error": float(err),
                "basis_probes": {str(radius): [central(v, radius) for v in np.eye(DIM)] for radius in RADII},
            }
        fit_j = np.stack([(np.array(r["plus"]) - r["minus"]) / (2 * FIT) for r in previous["fit"]], axis=1)
        norms = np.linalg.norm(basis.reshape(DIM, -1), axis=1)
        keep = norms >= 1
        normalised_j = fit_j[:, keep] / norms[keep]
        _, _, vectors = np.linalg.svd(normalised_j, full_matrices=True)
        contrasts = {}
        for label, u in [("most_sensitive", vectors[0]), ("least_sensitive", vectors[-1])]:
            v = np.zeros(DIM)
            v[keep] = u / norms[keep]
            contrasts[label] = [central(v, eps) for eps in RADII]
        for axis in range(3):
            for label, signs in [("same", [1, 1, 1, 1]), ("alternating", [1, -1, 1, -1])]:
                v = np.zeros((4, 3))
                v[:, axis] = np.array(signs) / 2
                contrasts[f"{axis}_{label}"] = [central(v.flatten(), FIT)]
        zero, cc = rollout(env, base, ids, robot_geoms)
        if not np.allclose(zero, y0, rtol=0, atol=1e-9) or cc != c0:
            raise ValueError("Second pass replay mismatch")
        return {"contrasts": contrasts, "retained_input_dim": int(keep.sum())}

    fit = [central(v, FIT) for v in np.eye(DIM)]
    random = rng.normal(size=(12, DIM))
    random /= np.linalg.norm(random, axis=1, keepdims=True)
    tests = [central(v, eps) for eps in RADII for v in random]
    repeat, repeat_contacts = rollout(env, base, ids, robot_geoms)
    repeat_error = float(np.max(np.abs(repeat - y0)))
    if repeat_error > 1e-9 or repeat_contacts != c0:
        raise ValueError(f"Unperturbed replay mismatch: {repeat_error}")
    return {
        "start": t,
        "eef_restore_error_m": float(pose_error),
        "body_names": names,
        "replay_error": repeat_error,
        "base_contact_steps": sum(bool(c) for c in c0),
        "basis": basis.tolist(),
        "fit": fit,
        "tests": tests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fit-root", type=Path, help="Fit records for independent targeted probes")
    parser.add_argument("--scale-check", action="store_true", help="Re-estimate each Jacobian at three radii")
    args = parser.parse_args()
    if args.scale_check and not args.fit_root:
        parser.error("--scale-check requires --fit-root")
    import libero.libero as libero
    import robosuite
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs.env_wrapper import ControlEnv

    libero._assets_path_cache = str(args.assets)
    suite = get_benchmark_dict()["libero_10"]()
    (task_id,) = [i for i, task in enumerate(suite.tasks) if task.name + "_demo" == args.file.stem]
    task = suite.get_task(task_id)
    env = ControlEnv(
        bddl_file_name=str(Path(libero.get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        use_camera_obs=False,
        has_offscreen_renderer=False,
        camera_names=[],
        ignore_done=True,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(args.file) as source:
            for name in ("demo_0", "demo_1"):
                demo = source[f"data/{name}"]
                times = anchors(demo["actions"][:])
                for phase, t in zip(("before", "switch", "after"), times, strict=True):
                    dest = args.out / f"{task.name}--{name}--{phase}.json"
                    if dest.exists():
                        continue
                    env.reset()
                    xml = relocate(
                        demo.attrs["model_file"],
                        args.assets,
                        Path(robosuite.__file__).parent / "models/assets",
                    )
                    env.reset_from_xml_string(xml)
                    rng = np.random.default_rng(1400 + t + int(name[-1]) * 1000 + task_id * 10000)
                    previous = json.loads((args.fit_root / dest.name).read_text()) if args.fit_root else None
                    row = probe(env, demo, t, rng, previous, args.scale_check)
                    row.update(task=task.name, demo=name, phase=phase)
                    tmp = dest.with_suffix(".tmp")
                    tmp.write_text(json.dumps(row, separators=(",", ":"), allow_nan=False))
                    tmp.replace(dest)
    finally:
        env.close()


if __name__ == "__main__":
    main()
