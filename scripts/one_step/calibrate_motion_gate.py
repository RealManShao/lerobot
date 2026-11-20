"""Activate the history servo only when measured motion advances along the visual waypoint."""

import argparse
import json
from pathlib import Path

import torch
from chunk_benchmark import dataset, fit_mask
from train_history_control import samples

from lerobot.policies.demonstration_chunk.hybrid_control import HistoryControlHead


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "cache", "suite"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    report = json.loads((a.root / "history_adaptive.json").read_text())
    if report["seed"] != a.seed:
        raise ValueError("Expert seed mismatch")
    model = {**report["model"], "mode": "history_aligned"}
    head = HistoryControlHead(**model).cuda().eval()
    inc = head.load_state_dict(
        torch.load(a.root / f"history_adaptive-seed{a.seed}.pt", weights_only=True), strict=False
    )
    if inc.missing_keys != ["goal_noise"] or inc.unexpected_keys:
        raise ValueError(inc)
    for split in ("train", "test"):
        data, records = dataset(a.cache, split, 10)
        idx = (
            (~fit_mask(data, records, 8)).nonzero().flatten()
            if split == "train"
            else torch.arange(len(data[0]), device="cuda")
        )
        xyz = samples(records, a.suite)[idx]
        state = data[1][idx]
        torch.testing.assert_close(state[:, :3], xyz[:, 2])
        past = torch.stack([xyz[:, 2] - xyz[:, 1], xyz[:, 1] - xyz[:, 0]], 1)
        blocks = torch.arange(len(idx), device="cuda").split(128)
        part = [head.adjusted(data[0][idx[b]], state[b], past[b]) for b in blocks]
        actions, goal, feature = (torch.cat(x) for x in zip(*part, strict=True))
        if split == "train":
            head.goal_noise.copy_((goal[:, 0] - xyz[:, -1]).square().sum(1).mean().sqrt())
            head.constant_weight.copy_(head.alpha(feature).mean())
        else:
            servo = (
                ((goal[:, :1] - xyz[:, 2:7]) @ head.inverse_motion)
                / torch.arange(5, 0, -1, device="cuda")[None, :, None]
            ).clamp(-1, 1)
            for name in head.alignment_modes:
                head.mode = name
                w = head.alpha(feature)
                pred = torch.lerp(actions[:, :5, :3].clamp(-1, 1), servo, w[:, None, None])
                doc = {
                    **report,
                    "model": {**model, "mode": name},
                    "goal_noise_m": head.goal_noise.item(),
                    "constant_weight": head.constant_weight.item(),
                    "weight_mean": w.mean().item(),
                    "weight_std": w.std().item(),
                    "tracking_mse": (pred - data[2][idx, :5, :3]).square().mean().item(),
                }
                torch.save(head.state_dict(), a.root / f"{name}-seed{a.seed}.pt")
                (a.root / f"{name}.json").write_text(json.dumps(doc, indent=2) + "\n")


if __name__ == "__main__":
    main()
