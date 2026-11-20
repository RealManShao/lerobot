"""Paired offline DrifOv boundary-loss ablation on one raw LIBERO task.

Uses a small LeRobot action head with fixed RGB mean-pooling features (no pretrained VLM).
Inputs are native HDF5 actions, 8D end-effector/gripper state, and both camera views.
Splits by demonstration before making windows. Metrics are offline action errors.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from lerobot.policies.drif_ov.configuration_drif_ov import DrifOvConfig
from lerobot.policies.drif_ov.modeling_drif_ov import ACTION_PREFIX, PREFIX_VALID_MASK, DrifOvActionHead

HORIZON = 40
BATCH_SIZE = 32


def load_windows(group, names, device):
    cameras, states, actions = [], [], []
    for name in names:
        episode = group[name]
        obs = episode["obs"]
        action = episode["actions"][:]
        starts = np.arange(0, len(action) - HORIZON + 1, 8)
        if len(starts) == 0:
            continue
        views = []
        for camera in ("agentview_rgb", "eye_in_hand_rgb"):
            images = obs[camera][: starts[-1] + 1 : 8]
            n, h, w, channels = images.shape
            pooled = images.reshape(n, 4, h // 4, 4, w // 4, channels).mean(axis=(2, 4))
            views.append(pooled.reshape(n, -1) / 255.0)
        cameras.append(np.stack(views, axis=1))
        states.append(
            np.concatenate([obs[k][:][starts] for k in ("ee_pos", "ee_ori", "gripper_states")], axis=1)
        )
        actions.append(action[starts[:, None] + np.arange(HORIZON)])
    return [
        torch.as_tensor(np.concatenate(x), dtype=torch.float32, device=device)
        for x in (cameras, states, actions)
    ]


def inputs(data, indices):
    cameras, state, action = [x[indices] for x in data]
    valid = torch.ones(cameras.shape[:2], device=state.device, dtype=torch.bool)
    features = {"backbone_features": cameras, "image_mask": valid, "backbone_attention_mask": valid}
    batch = {
        "state": state[:, None],
        "action": action,
        "action_mask": torch.ones_like(action),
        "embodiment_id": torch.zeros(len(state), device=state.device, dtype=torch.long),
    }
    return features, batch


@torch.inference_mode()
def evaluate(head, data):
    head.eval()
    device = data[0].device
    rows = []
    for length in (0, 5, 20):
        for noisy in (False, True) if length else (False,):
            generator = torch.Generator(device=device).manual_seed(2026)
            errors = []
            for indices in torch.arange(len(data[0]), device=device).split(BATCH_SIZE):
                features, batch = inputs(data, indices)
                target = batch.pop("action")
                prefix = target.clone()
                mask = torch.zeros_like(prefix, dtype=torch.bool)
                mask[:, :length] = True
                if noisy:
                    noise = torch.randn(prefix.shape, device=device, generator=generator)
                    prefix += (noise * 0.02).clamp(-0.05, 0.05) * mask
                batch.update({ACTION_PREFIX: prefix, PREFIX_VALID_MASK: mask})
                pred = head.get_action(features, batch)["action_pred"]
                delta = (pred - target).square()
                errors.append(
                    torch.stack(
                        (
                            delta[:, length, :6].mean(-1),
                            delta[:, length, 6],
                            delta[:, length:, :6].mean((1, 2)),
                        ),
                        dim=1,
                    )
                )
            arm, gripper, suffix = torch.cat(errors).mean(0).tolist()
            rows.append(
                {
                    "length": length,
                    "noisy": noisy,
                    "first_arm_mse": arm,
                    "first_gripper_mse": gripper,
                    "suffix_arm_mse": suffix,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 42, 1000])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    with h5py.File(args.task) as source:
        group = source["data"]
        names = sorted(group, key=lambda name: int(name.split("_")[-1]))
        order = np.random.default_rng(2026).permutation(names).tolist()
        cut = int(0.8 * len(order))
        train = load_windows(group, order[:cut], device)
        test = load_windows(group, order[cut:], device)

    report = {
        "task": args.task.name,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "steps": args.steps,
        "train_demos": order[:cut],
        "test_demos": order[cut:],
        "train_windows": len(train[0]),
        "test_windows": len(test[0]),
        "results": [],
    }
    for seed in args.seeds:
        for name, action_weight, velocity_weight in (("original", 1.0, 0.1), ("action_only", 1.1, 0.0)):
            torch.manual_seed(seed)
            config = DrifOvConfig(
                device="cuda",
                chunk_size=HORIZON,
                n_action_steps=HORIZON,
                max_state_dim=train[1].shape[-1],
                max_action_dim=train[2].shape[-1],
                max_num_embodiments=1,
                backbone_embedding_dim=train[0].shape[-1],
                action_model_dim=64,
                action_model_num_layers=2,
                action_model_num_heads=4,
                action_model_ff_dim=128,
                use_bf16=False,
                boundary_action_loss_weight=action_weight,
                boundary_velocity_loss_weight=velocity_weight,
            )
            head = DrifOvActionHead(config).to(device).train()
            optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4)
            for _ in range(args.steps):
                indices = torch.randint(len(train[0]), (BATCH_SIZE,), device=device)
                features, batch = inputs(train, indices)
                optimizer.zero_grad(set_to_none=True)
                head(features, batch)["loss"].backward()
                optimizer.step()
            scores = evaluate(head, test)
            report["results"].append({"seed": seed, "variant": name, "metrics": scores})
            print(seed, name, scores[1], flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
