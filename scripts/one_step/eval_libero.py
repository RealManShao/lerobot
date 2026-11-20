"""Run a paired LIBERO-Long pilot for a trained one-step head (K=5)."""

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from benchmark_libero_feedback import _make_environment, _prepare_libero_resources, observation_fingerprint
from cache_libero import HORIZON, TASKS, encode_views
from transformers import Qwen2VLImageProcessor

from lerobot.policies.demonstration_chunk.waypoint import waypoint_translation
from lerobot.policies.groot.modeling_groot import GrootPolicy


def transform(image, code):
    return image[:: -1 if code & 1 else 1, :: -1 if code & 2 else 1]


def robot_state(obs):
    state = obs["robot_state"]
    from robosuite.utils.transform_utils import quat2axisangle

    rot = quat2axisangle(state["eef"]["quat"].copy())
    return np.concatenate([state["eef"]["pos"], rot, state["gripper"]["qpos"]])


def calibrate(env, episode):
    errors = np.zeros((2, 4))
    state_errors = []
    for t in (0, 40, 80):
        obs = env._format_raw_obs(env._env.set_init_state(episode["states"][t + 1]))
        target = np.concatenate([episode["obs"][k][t] for k in ("ee_pos", "ee_ori", "gripper_states")])
        state_errors.append(float(np.max(np.abs(robot_state(obs) - target))))
        for view, (camera, key) in enumerate((("image", "agentview_rgb"), ("image2", "eye_in_hand_rgb"))):
            reference = episode["obs"][key][t].astype(float)
            for code in range(4):
                errors[view, code] += np.abs(transform(obs["pixels"][camera], code) - reference).mean() / 3
    return {
        "sim_state_offset": 1,
        "flips": errors.argmin(1).tolist(),
        "rgb_mae_0_255": errors.min(1).tolist(),
        "max_state_abs_diff": max(state_errors),
    }


@torch.inference_mode()
def plan(obs, camera_flips, task_id, head, visual, processor, n_tasks=len(TASKS), history=None):
    images = {}
    for key, camera, flip in zip(
        ("agentview_rgb", "eye_in_hand_rgb"), ("image", "image2"), camera_flips, strict=True
    ):
        images[key] = transform(obs["pixels"][camera], flip)[None]
    features = encode_views(images, np.array([0]), visual, processor).cuda().float()
    mask = torch.ones(features.shape[:2], dtype=torch.bool, device="cuda")
    state = robot_state(obs)
    if task_id is not None:
        state = np.concatenate([state, np.eye(n_tasks)[task_id]])
    batch = {
        "embodiment_id": torch.zeros(1, dtype=torch.long, device="cuda"),
        "state": torch.as_tensor(state, dtype=torch.float32, device="cuda")[None, None],
    }
    if history is not None:
        batch["displacements"] = torch.as_tensor(history, dtype=torch.float32, device="cuda")[None]
    output = head.get_action(
        {"backbone_features": features, "backbone_attention_mask": mask, "image_mask": mask}, batch
    )
    actions = output["action_pred"][0].clamp(-1, 1)
    actions[:, 6] = torch.where(actions[:, 6] > 0, 1.0, -1.0)
    if "weight" in output:
        controller = {
            "goal": output["goals"][0, 0].cpu().numpy(),
            "weight": output["weight"][0, 0].item(),
            "learned": output["learned_xyz"][0, :5].clamp(-1, 1).cpu().numpy(),
        }
        if "estimated_contact" in output:
            controller["geometry"] = (
                output["positions"][0].cpu().numpy(),
                output["rotations"][0].cpu().numpy(),
                output["estimated_contact"][0].item(),
            )
    else:
        controller = None
    return actions.cpu().numpy(), controller


class Transplant:
    """Counterfactual actuator swap between two already trained heads."""

    def __init__(self, arm, gripper):
        self.arm = arm
        self.gripper = gripper

    def get_action(self, features, batch):
        arm = self.arm.get_action(features, batch)["action_pred"]
        grip = self.gripper.get_action(features, batch)["action_pred"]
        return {"action_pred": torch.cat([arm[..., :6], grip[..., 6:]], -1)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "out", "suite", "checkpoint"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--libero-assets", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--gripper-variant", help="Use the gripper from a second demonstration-chunk checkpoint")
    p.add_argument(
        "--fixed-weight",
        type=float,
        choices=[0.0, 0.5, 1.0],
        help="Override a constant router for a matched blend control",
    )
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--first-state", type=int, default=0)
    p.add_argument("--task-ids", type=int, nargs="+")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _prepare_libero_resources(args, args.out)
    from lerobot.envs.libero import _get_suite

    suite = _get_suite("libero_10")
    policy = GrootPolicy.from_pretrained(str(args.checkpoint), local_files_only=True)
    visual = policy._groot_model.backbone.visual.to(device="cuda", dtype=torch.bfloat16).eval()
    del policy
    processor = Qwen2VLImageProcessor(
        patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3
    )
    run = json.loads((args.root / f"{args.variant}.json").read_text())
    task_names = run.get("task_names", TASKS)
    task_blind = run.get("policy_type") == "visual_geometry"
    if task_blind:
        from label_geometry import entities

        from lerobot.policies.demonstration_chunk.visual_geometry import VisualGeometryHead

        if args.seed != run["seed"]:
            raise ValueError("Checkpoint and report seeds differ")
        head = VisualGeometryHead(**run["model"]).cuda().eval()
    elif run.get("policy_type") in {"hybrid_control", "outcome_control"}:
        from lerobot.policies.demonstration_chunk.hybrid_control import HybridControlHead, OutcomeControlHead

        if args.seed != run["seed"]:
            raise ValueError("Checkpoint and report seeds differ")
        head_type = OutcomeControlHead if run["policy_type"] == "outcome_control" else HybridControlHead
        head = head_type(**run["model"]).cuda().eval()
    elif run.get("policy_type") == "history_control":
        from lerobot.policies.demonstration_chunk.hybrid_control import HistoryControlHead

        if run["seed"] != args.seed:
            raise ValueError("Checkpoint and report seeds differ")
        head = HistoryControlHead(**run["model"]).cuda().eval()
    elif run.get("policy_type") == "gripper_control":
        from lerobot.policies.demonstration_chunk.gripper_control import GripperControlHead

        if args.seed != run["seed"]:
            raise ValueError("Checkpoint and report seeds differ")
        head = GripperControlHead(**run["model"]).cuda().eval()
    elif run.get("policy_type") == "demonstration_chunk":
        from lerobot.policies.demonstration_chunk.modeling_demonstration_chunk import DemonstrationChunkHead

        if run["seed"] != args.seed:
            raise ValueError(f"Report is for seed {run['seed']}, requested {args.seed}")
        head = DemonstrationChunkHead(**run["model"]).cuda().eval()
    else:
        from lerobot.policies.drif_ov import DrifOvConfig
        from lerobot.policies.drif_ov.modeling_drif_ov import DrifOvActionHead

        cfg = DrifOvConfig(
            device="cuda",
            chunk_size=HORIZON,
            n_action_steps=HORIZON,
            max_state_dim=11,
            max_action_dim=7,
            max_num_embodiments=1,
            backbone_embedding_dim=2048,
            action_model_dim=run["width"],
            action_model_num_layers=run["layers"],
            action_model_num_heads=4,
            action_model_ff_dim=2 * run["width"],
            use_bf16=False,
        )
        head = DrifOvActionHead(cfg).cuda().eval()
    weights = run["weights"] if task_blind else f"{args.variant}-seed{args.seed}.pt"
    head.load_state_dict(torch.load(args.root / weights, weights_only=True))
    if args.fixed_weight is not None:
        if run.get("policy_type") != "hybrid_control" or head.mode != "constant":
            raise ValueError("--fixed-weight requires the constant hybrid checkpoint")
        head.constant_weight.fill_(args.fixed_weight)
    if args.gripper_variant:
        if run.get("policy_type") != "demonstration_chunk":
            raise ValueError("Gripper transplant requires a demonstration-chunk policy")
        grip_run = json.loads((args.root / f"{args.gripper_variant}.json").read_text())
        if grip_run["task_names"] != task_names or grip_run["seed"] != args.seed:
            raise ValueError("Gripper transplant task order or training seed differs")
        gripper = DemonstrationChunkHead(**grip_run["model"]).cuda().eval()
        gripper.load_state_dict(
            torch.load(args.root / f"{args.gripper_variant}-seed{args.seed}.pt", weights_only=True)
        )
        head = Transplant(head, gripper)
    result = {
        "variant": args.variant,
        **({"fixed_weight": args.fixed_weight} if args.fixed_weight is not None else {}),
        **({"gripper_variant": args.gripper_variant} if args.gripper_variant else {}),
        "seed": args.seed,
        "K": 5,
        "H": HORIZON,
        "calibration": {},
        "episodes": [],
    }
    needs_history = run.get("policy_type") == "history_control"
    hybrid = needs_history or task_blind or run.get("policy_type") in {"hybrid_control", "outcome_control"}
    if task_blind:
        result["task_id_input"] = False
        result["training_steps"] = run["steps"]
    if hybrid:
        inverse_motion = head.inverse_motion.cpu().numpy()
        result["goal_predicates"] = {}
    for i in args.task_ids if args.task_ids is not None else range(len(task_names)):
        prefix = task_names[i]
        (task,) = [j for j, item in enumerate(suite.tasks) if item.name.startswith(prefix)]
        (path,) = args.suite.glob(f"{prefix}*.hdf5")
        calibration_env = _make_environment(suite, task, 520, 128)
        try:
            calibration_env.reset(seed=123)
            with h5py.File(path) as f:
                cal = calibrate(calibration_env, f["data/demo_0"])
            result["calibration"][prefix] = cal
            if cal["max_state_abs_diff"] > 0.05 or max(cal["rgb_mae_0_255"]) > 30:
                raise RuntimeError(f"Training observations could not be aligned for {prefix}: {cal}")
        finally:
            calibration_env.close()
        for n in range(args.first_state, args.first_state + args.episodes):
            rng = 9000 + n
            random.seed(rng)
            np.random.seed(rng)
            torch.manual_seed(rng)
            env = _make_environment(suite, task, 520, 128)
            try:
                env.init_state_id = n
                obs, _ = env.reset(seed=rng)
                fingerprint = observation_fingerprint(obs)
                if hybrid:
                    domain = env._env.env
                    predicates = domain.parsed_problem["goal_state"]
                    result["goal_predicates"][prefix] = predicates
                history = np.zeros((2, 3)) if needs_history else None
                previous_pos = obs["robot_state"]["eef"]["pos"].copy() if needs_history else None
                if task_blind:
                    if cal["flips"] != [0, 0]:
                        raise ValueError("Unexpected camera orientation for the fixed task-blind pipeline")
                    _, slots, fingers = entities(domain)
                    perception = np.zeros(7)
                trace = []
                steps = 0
                done = success = False
                while steps < 520 and not done:
                    actions, controller = plan(
                        obs,
                        [0, 0] if task_blind else cal["flips"],
                        None if task_blind else i,
                        head,
                        visual,
                        processor,
                        len(task_names),
                        history,
                    )
                    if controller is not None:
                        bits = sum(
                            int(bool(domain._eval_predicate(p))) << j for j, p in enumerate(predicates)
                        )
                        trace.append([bits, round(controller["weight"], 4), int(actions[0, 6])])
                    if task_blind:
                        # Diagnostics after planning; simulator geometry never enters the policy.
                        pos_hat, rot_hat, touch_hat = controller["geometry"]
                        data = domain.sim.data
                        touched = set()
                        for c in data.contact[: data.ncon]:
                            if c.geom1 in fingers:
                                touched.add(c.geom2)
                            if c.geom2 in fingers:
                                touched.add(c.geom1)
                        truth = any(geoms & touched for _, _, geoms in slots)
                        for j, (kind, k, _) in enumerate(slots):
                            pos, rot = (
                                (data.site_xpos[k], data.site_xmat[k])
                                if kind == "site"
                                else (data.body_xpos[k], data.body_xmat[k])
                            )
                            perception[0] += np.linalg.norm(pos_hat[j] - pos)
                            perception[1] += np.rad2deg(
                                np.arccos(
                                    np.clip((np.trace(rot_hat[j].T @ rot.reshape(3, 3)) - 1) / 2, -1, 1)
                                )
                            )
                        perception[2:] += [len(slots), truth, touch_hat >= 0.5, truth and touch_hat >= 0.5, 1]
                    for tick, action in enumerate(actions[:5]):
                        if controller is not None:
                            pos = obs["robot_state"]["eef"]["pos"]
                            servo = waypoint_translation(pos, controller["goal"], inverse_motion, 5 - tick)
                            learned = controller["learned"][tick]
                            action[:3] = learned + controller["weight"] * (servo - learned)
                        obs, _, terminated, truncated, info = env.step(action)
                        if needs_history:
                            new_pos = obs["robot_state"]["eef"]["pos"].copy()
                            history[1] = history[0]
                            history[0] = new_pos - previous_pos
                            previous_pos = new_pos
                        steps += 1
                        success = bool(info["is_success"])
                        done = success or terminated or truncated or steps >= 520
                        if done:
                            break
                result["episodes"].append(
                    {
                        "task": prefix,
                        "initial_state": n,
                        "fingerprint": fingerprint,
                        "steps": steps,
                        "success": success,
                        **({"gate_trace": trace} if hybrid else {}),
                        **(
                            {
                                "perception_totals": dict(
                                    zip(
                                        (
                                            "position_m",
                                            "angle_deg",
                                            "slots",
                                            "contact",
                                            "called_contact",
                                            "true_positive",
                                            "plans",
                                        ),
                                        perception.tolist(),
                                        strict=True,
                                    )
                                )
                            }
                            if task_blind
                            else {}
                        ),
                    }
                )
                output = args.out / "episodes.json"
                tmp = output.with_suffix(".tmp")
                tmp.write_text(json.dumps(result, indent=2) + "\n")
                tmp.replace(output)
            finally:
                env.close()


if __name__ == "__main__":
    main()
