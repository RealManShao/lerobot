"""Fit a causal waypoint correction and a router on disjoint demonstration episodes."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from chunk_benchmark import dataset, fit_mask
from train_hybrid_control import optimal_weight

from lerobot.policies.demonstration_chunk.hybrid_control import HistoryControlHead


def samples(records, suite):
    rows = []
    for r in records:
        with h5py.File(suite / (r["task"] + ".hdf5")) as f:
            for demo in r["demos"]:
                xyz = f["data"][demo]["obs/ee_pos"][:]
                starts = np.arange(0, len(xyz) - 40, 8)
                rows.append(xyz[np.maximum(starts[:, None] + np.arange(-2, 6), 0)])
    return torch.as_tensor(np.concatenate(rows), device="cuda").float()


def rms_mm(error):
    return error.square().sum(-1).mean().sqrt().item() * 1000


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "cache", "suite"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    base = json.loads((a.root / "adaptive.json").read_text())
    if base["seed"] != a.seed:
        raise ValueError("Seed mismatch")
    model = {"expert_config": base["model"]["expert_config"], "mode": "history_adaptive"}
    head = HistoryControlHead(**model).cuda()
    inc = head.load_state_dict(
        torch.load(a.root / f"adaptive-seed{a.seed}.pt", weights_only=True), strict=False
    )
    if inc.unexpected_keys or any(
        not k.startswith(("history_fit", "history_gate.")) for k in inc.missing_keys
    ):
        raise ValueError(inc)
    opt = torch.optim.AdamW(head.history_gate.parameters(), lr=5e-4)
    for split in ("train", "test"):
        data, records = dataset(a.cache, split, 10)
        xyz = samples(records, a.suite)
        state = data[1]
        torch.testing.assert_close(state[:, :3], xyz[:, 2])
        history = torch.stack([xyz[:, 2] - xyz[:, 1], xyz[:, 1] - xyz[:, 0]], dim=1)
        target = xyz[:, -1] - xyz[:, 2]
        chunks = torch.arange(len(state), device="cuda").split(128)
        with torch.no_grad():
            old = torch.cat([head.goals(data[0][b], state[b])[1][:, 0] for b in chunks])
            design = torch.cat([old, history.flatten(1)], 1)
            if split == "train":
                mask = fit_mask(data, records, 8)
                head.history_fit.copy_(torch.linalg.lstsq(design[mask], target[mask]).solution)
            parts = [head.adjusted(data[0][b], state[b], history[b]) for b in chunks]
            actions, goals, features = (torch.cat(x) for x in zip(*parts, strict=True))
            learned = actions[:, :5, :3].clamp(-1, 1)
            servo = (
                ((goals[:, :1] - xyz[:, 2:7]) @ head.inverse_motion)
                / torch.arange(5, 0, -1, device="cuda")[None, :, None]
            ).clamp(-1, 1)
            delta = servo - learned
            res = data[2][:, :5, :3] - learned
        if split == "train":
            calib = (~mask).nonzero().flatten()
            head.constant_weight.copy_(optimal_weight(delta[calib], res[calib]))
            for _ in range(350):
                ids = calib[torch.randint(len(calib), (128,), device="cuda")]
                w = head.alpha(features[ids])
                loss = (res[ids] - w[:, None, None] * delta[ids]).square().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        else:
            with torch.no_grad():
                for mode in head.history_modes:
                    head.mode = mode
                    w = head.alpha(features)
                    report = {
                        "model": {**model, "mode": mode},
                        "policy_type": "history_control",
                        "task_names": base["task_names"],
                        "seed": a.seed,
                        "action_offset": 1,
                        "old_waypoint_rmse_mm": rms_mm(old - target),
                        "ballistic_rmse_mm": rms_mm(5 * history[:, 0] - target),
                        "waypoint_rmse_mm": rms_mm(goals[:, 0] - xyz[:, -1]),
                        "weight_mean": w.mean().item(),
                        "weight_std": w.std().item(),
                        "tracking_mse": (res - w[:, None, None] * delta).square().mean().item(),
                        "constant_weight": head.constant_weight.item(),
                    }
                    torch.save(head.state_dict(), a.root / f"{mode}-seed{a.seed}.pt")
                    (a.root / f"{mode}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
