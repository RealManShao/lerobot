"""Separate fit quality, held-out sensor use and moving-entity errors over training."""

import argparse
import json
from pathlib import Path

import torch
from chunk_benchmark import dataset, fit_mask
from train_visual_geometry import collect, labels

from lerobot.policies.demonstration_chunk.visual_geometry import VisualGeometryHead, rotation


def swapped_windows(data):
    """Swap cameras with a different held-out episode at the same relative episode time."""
    other = torch.empty(len(data[0]), device="cuda", dtype=torch.long)
    for t in range(10):
        task = data[3] == t
        eps = data[4][task].unique()
        for i, e in enumerate(eps):
            dest = (task & (data[4] == e)).nonzero().flatten()
            src = (task & (data[4] == eps[(i + 1) % len(eps)])).nonzero().flatten()
            aligned = torch.linspace(0, len(src) - 1, len(dest), device="cuda").round().long()
            other[dest] = src[aligned]
    if (data[4][other] == data[4]).any():
        raise ValueError("Paired image swap failed")
    return other


def describe(head, data, lab, idx):
    arm, _, goal, pos, rot, _, _, _, pr, _ = collect(head, data, idx)
    pose, touch, end, valid = (x[idx] for x in lab)
    distance = (pos - pose[..., :3]).norm(dim=-1)
    angle = (
        (((rotation(pose[..., 3:]).transpose(-1, -2) @ rot).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2)
        .clamp(-1, 1)
        .acos()
        .rad2deg()
    )
    gt = touch.bool().any(1)
    pred = pr >= 0.5
    sorted_truth = gt[pr.argsort(descending=True)].float()
    precision = sorted_truth.cumsum(0) / torch.arange(1, len(idx) + 1, device="cuda")
    moving = torch.zeros_like(valid)
    for i in range(10):
        task = data[3][idx] == i
        for e in data[4][idx][task].unique():
            mask = task & (data[4][idx] == e)
            loc = pose[mask, :, :3]
            travel = (loc - loc[:1]).norm(dim=-1).max(0).values
            moving[mask] = travel > 0.02
    return {
        "windows": len(idx),
        "arm5_mse": (arm[:, :5] - data[2][idx, :5, :6]).square().mean().item(),
        "pose_mean_mm": 1000 * distance[valid].mean().item(),
        "moving_pose_mean_mm": 1000 * distance[moving & valid].mean().item(),
        "moving_rows": (moving & valid).sum().item(),
        "angle_deg": angle[valid].mean().item(),
        "goal_rms_mm": 1000 * (goal - end).square().sum(1).mean().sqrt().item(),
        "contact_prevalence": gt.float().mean().item(),
        "contact_f1": (2 * (gt & pred).sum() / (gt.sum() + pred.sum()).clamp_min(1)).item(),
        "contact_average_precision": (
            (precision * sorted_truth).sum() / sorted_truth.sum().clamp_min(1)
        ).item(),
    }


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "cache", "labels", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--geometry", choices=["object", "world", "consistent", "shared"], required=True)
    a = p.parse_args()
    train, records = dataset(a.cache, "train", 10)
    held, rec = dataset(a.cache, "test", 10)
    tr = labels(a.labels, "train", train, records)
    te = labels(a.labels, "test", held, rec)
    mask = fit_mask(train, records, 8)
    swapped = swapped_windows(held)
    mismatch = (held[0][swapped], *held[1:])
    ids = torch.arange(len(held[0]), device="cuda")
    result = {}
    for step in (1800, 7200, 20000):
        root = a.root / f"step{step}" / f"seed{a.seed}"
        r = json.loads((root / f"{a.geometry}_direct.json").read_text())
        head = VisualGeometryHead(**r["model"]).cuda().eval()
        head.load_state_dict(torch.load(root / r["weights"], weights_only=True))
        result[step] = {
            "fit": describe(head, train, tr, mask.nonzero().flatten()),
            "calibration": describe(head, train, tr, (~mask).nonzero().flatten()),
            "test": describe(head, held, te, ids),
            "swapped_cameras": describe(head, mismatch, te, ids),
        }
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
