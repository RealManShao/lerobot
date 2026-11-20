"""Train task-blind visual policies and checkpoint a fixed long-budget optimization trajectory."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from chunk_benchmark import dataset, fit_mask
from torch.nn import functional

from lerobot.policies.demonstration_chunk.visual_geometry import VisualGeometryHead, rotation


def labels(folder, split, data, records):
    labels = []
    for i, r in enumerate(records):
        with np.load(folder / f"{i}-{split}.npz") as z:
            if int(z["label_version"]) != 2:
                raise ValueError("Expected contact labels resolved by MuJoCo geometry ID")
            task = data[3] == i
            ep = torch.as_tensor(z["episode"], device="cuda")
            torch.testing.assert_close(ep, data[4][task])
            if str(z["task"]) != r["task"] or r["action_offset"] != 1:
                raise ValueError("Task/timing mismatch in labels")
            for number in ep.unique():
                mask = (ep == number).cpu().numpy()
                np.testing.assert_equal(z["start"][mask], np.arange(mask.sum()) * 8)
            inputs = [torch.as_tensor(z[k], device="cuda") for k in ("pose", "touch", "goal")]
            valid = torch.arange(4, device="cuda")[None].expand(len(ep), -1) < len(z["names"])
            labels.append((*inputs, valid))
    return tuple(torch.cat(x) for x in zip(*labels, strict=True))


def losses(head, pred, labels, target):
    pose, touch, end, mask = labels
    arm, grip, goal, pos, rot, valid, logit, assoc, _, local = pred
    true_rot = rotation(pose[..., 3:])
    weight = mask.float()

    def masked(x):
        return (x * weight).sum() / weight.sum()

    nearest = (pose[..., :3] - end[:, None]).square().sum(-1).masked_fill(~mask, torch.inf).argmin(1)
    loss = 5 * functional.mse_loss(arm, target[..., :6]) + functional.binary_cross_entropy_with_logits(
        grip, (target[..., 6] > 0).float()
    )
    loss = loss + masked((pos - pose[..., :3]).square().mean(-1)) / 0.035**2
    loss = (
        loss
        + masked((rot - true_rot).square().mean((-1, -2)))
        + functional.binary_cross_entropy_with_logits(valid, weight)
    )
    loss = (
        loss
        + masked(functional.binary_cross_entropy_with_logits(logit, touch, reduction="none"))
        + 0.3 * functional.cross_entropy(assoc, nearest)
    )
    waypoint_loss = functional.smooth_l1_loss((goal - end) / 0.025, torch.zeros_like(end))
    if head.geometry == "consistent":
        measured = pose[..., :3] + (true_rot @ local[..., None]).squeeze(-1)
        estimated = pos + (rot @ local[..., None]).squeeze(-1)
        candidates = torch.where(mask[..., None], measured, estimated)
        mixed_goal = (assoc.softmax(1)[..., None] * candidates).sum(1)
        substituted = functional.smooth_l1_loss((mixed_goal - end) / 0.025, torch.zeros_like(end))
        waypoint_loss = 0.5 * (waypoint_loss + substituted)
    loss = loss + waypoint_loss
    if head.geometry in {"object", "consistent"}:
        offsets = (true_rot.transpose(-1, -2) @ (end[:, None] - pose[..., :3])[..., None]).squeeze(-1)
        loss = loss + 0.2 * masked(
            functional.smooth_l1_loss(local / 0.06, offsets / 0.06, reduction="none").mean(-1)
        )
    return loss


@torch.inference_mode()
def collect(head, data, ids):
    chunks = [head.predict(data[0][b], data[1][b, :8]) for b in ids.split(128)]
    return tuple(torch.cat(v) for v in zip(*chunks, strict=True))


def translation_alternatives(head, arm, goal, current_position):
    """Five-step displacements under the fitted Cartesian dynamics, in meters."""
    learned = arm[:, :5, :3].clamp(-1, 1).sum(1) @ head.motion
    servo = ((goal - current_position) @ head.inverse_motion / 5).clamp(-1, 1) * 5 @ head.motion
    return learned, servo


@torch.inference_mode()
def calibrate(head, data, lab, ids):
    arm, _, goal, *rest = collect(head, data, ids)
    touch = rest[-2]
    learned, servo = translation_alternatives(head, arm, goal, data[1][ids, :3])
    residual = lab[2][ids] - data[1][ids, :3] - learned
    delta = servo - learned
    head.global_w.copy_(((delta * residual).sum() / delta.square().sum().clamp_min(1e-12)).clamp(0, 1))
    # Solve a bounded two-coefficient least-squares problem on a 0.025 grid.
    design = torch.stack([delta * (1 - touch[:, None]), delta * touch[:, None]], -1).reshape(-1, 2)
    grid = torch.cartesian_prod(*[torch.linspace(0, 1, 41, device="cuda")] * 2)
    gram = design.T @ design
    rhs = design.T @ residual.flatten()
    score = ((grid @ gram) * grid).sum(1) - 2 * (grid @ rhs)
    head.stage_w.copy_(grid[score.argmin()])


@torch.inference_mode()
def metrics(head, data, lab, ids):
    arm, grip, goal, pos, rot, valid, logit, assoc, contact, _ = collect(head, data, ids)
    pose, touch, end, mask = (x[ids] for x in lab)
    error = (pos - pose[..., :3]).norm(dim=-1)
    rel = rot.transpose(-1, -2) @ rotation(pose[..., 3:])
    angle = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1).acos().rad2deg()
    actual = touch.bool().any(1)
    det = contact >= 0.5
    nearest = (pose[..., :3] - end[:, None]).square().sum(-1).masked_fill(~mask, torch.inf).argmin(1)
    result = []
    for task in sorted(data[3][ids].unique().tolist()):
        m = data[3][ids] == task
        active = mask & m[:, None]
        tp = (actual[m] & det[m]).sum().item()
        actual_n = actual[m].sum().item()
        det_n = det[m].sum().item()
        result.append(
            {
                "task": task,
                "windows": m.sum().item(),
                "pose_position_mm": 1000 * error[active].mean().item(),
                "rotation_deg": angle[active].mean().item(),
                "waypoint_rms_mm": 1000 * (goal[m] - end[m]).square().sum(1).mean().sqrt().item(),
                "arm5_mse": (arm[m, :5] - data[2][ids[m], :5, :6]).square().mean().item(),
                "grip_accuracy": ((grip[m] > 0) == (data[2][ids[m], :, 6] > 0)).float().mean().item(),
                "positive": actual_n,
                "predicted_positive": det_n,
                "true_positive": tp,
                "anchor_accuracy": (assoc[m].argmax(1) == nearest[m]).float().mean().item(),
            }
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "labels", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--geometry", choices=["object", "world", "consistent", "shared"], required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    torch.set_float32_matmul_precision("high")
    tr, records = dataset(a.cache, "train", 10)
    te, held = dataset(a.cache, "test", 10)
    labels_tr = labels(a.labels, "train", tr, records)
    labels_te = labels(a.labels, "test", te, held)
    fit = fit_mask(tr, records, 8)
    train = fit.nonzero().flatten()
    cal = (~fit).nonzero().flatten()
    head = VisualGeometryHead(a.geometry).cuda()
    head.center.copy_(labels_tr[0][fit][labels_tr[3][fit]][:, :3].mean(0))
    head.state_mean.copy_(tr[1][fit, :8].mean(0))
    head.state_std.copy_(tr[1][fit, :8].std(0).clamp_min(1e-3))
    cmd = tr[2][fit, :5, :3].sum(1)
    dist = labels_tr[2][fit] - tr[1][fit, :3]
    head.motion.copy_(torch.linalg.lstsq(cmd, dist).solution)
    head.inverse_motion.copy_(torch.linalg.pinv(head.motion))
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20000, eta_min=3e-5)
    g = torch.Generator(device="cuda").manual_seed(a.seed)
    for step in range(1, 20001):
        ids = train[torch.randint(len(train), (128,), device="cuda", generator=g)]
        pred = head.predict(tr[0][ids], tr[1][ids, :8])
        loss = losses(head, pred, [x[ids] for x in labels_tr], tr[2][ids])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step in (1800, 7200, 20000):
            calibrate(head, tr, labels_tr, cal)
            evals = metrics(head, te, labels_te, torch.arange(len(te[0]), device="cuda"))
            if not np.isfinite([x["waypoint_rms_mm"] for x in evals]).all():
                raise ValueError("Nonfinite waypoint prediction")
            root = a.out / f"step{step}" / f"seed{a.seed}"
            root.mkdir(parents=True, exist_ok=True)
            weights = f"{a.geometry}-seed{a.seed}.pt"
            torch.save(head.state_dict(), root / weights)
            for mode in head.routes:
                variant = f"{a.geometry}_{mode}"
                cfg = {
                    "policy_type": "visual_geometry",
                    "model": {"geometry": a.geometry, "route": mode},
                    "weights": weights,
                    "seed": a.seed,
                    "steps": step,
                    "batch": 128,
                    "action_offset": 1,
                    "fit_windows": len(train),
                    "cal_windows": len(cal),
                    "task_names": [x["task"].removesuffix("_demo") for x in records],
                    "per_task": evals,
                    "global_weight": head.global_w.item(),
                    "stage_weights": head.stage_w.tolist(),
                }
                tmp = root / f"{variant}.tmp"
                tmp.write_text(json.dumps(cfg, indent=2) + "\n")
                tmp.replace(root / f"{variant}.json")


if __name__ == "__main__":
    main()
