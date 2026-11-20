"""Identify a translation model, fit waypoint prediction and calibrate a frozen-expert router."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from chunk_benchmark import dataset, fit_mask, score

from lerobot.policies.demonstration_chunk.hybrid_control import HybridControlHead
from lerobot.policies.demonstration_chunk.waypoint import waypoint_translation


def positions(suite, records):
    chunks = []
    for record in records:
        if record["action_offset"] != 1:
            raise ValueError("Expected post-action camera frames with next-action labels")
        rows = []
        with h5py.File(suite / (record["task"] + ".hdf5")) as f:
            for demo in record["demos"]:
                xyz = f["data"][demo]["obs"]["ee_pos"][:]
                starts = np.arange(0, len(xyz) - 40, 8)
                rows.append(torch.from_numpy(xyz[starts[:, None] + np.arange(41)]).float())
        block = torch.cat(rows)
        torch.testing.assert_close(block[:, 0], record["data"][1][:, :3])
        chunks.append(block)
    return torch.cat(chunks).cuda()


def mse_summary(actual, estimated):
    return {
        "mse": (estimated - actual).square().mean().item(),
        "r2": (1 - (estimated - actual).square().sum() / (actual - actual.mean(0)).square().sum()).item(),
    }


def optimal_weight(delta, residual):
    return ((delta * residual).sum() / delta.square().sum().clamp_min(1e-12)).clamp(0, 1)


def teacher_servo(head, xyz, goals):
    remaining = torch.arange(5, 0, -1, device=xyz.device)[None, None, :, None]
    return waypoint_translation(
        xyz[:, :-1].unflatten(1, (8, 5)), goals[:, :, None], head.inverse_motion, remaining
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cache", "suite", "expert", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    run = json.loads((args.expert / "separate_prefix.json").read_text())
    if run["reserve_demos"] != 8 or run["seed"] != args.seed:
        raise ValueError("Expert must use the same 32/8 episode split and seed")
    head = HybridControlHead(run["model"], "adaptive").cuda()
    head.learner.load_state_dict(
        torch.load(args.expert / f"separate_prefix-seed{args.seed}.pt", weights_only=True)
    )
    head.learner.requires_grad_(False)
    train, records = dataset(args.cache, "train", 10)
    test, held_records = dataset(args.cache, "test", 10)
    xyz = positions(args.suite, records)
    held_xyz = positions(args.suite, held_records)
    fitting = fit_mask(train, records, 8)
    fit = fitting.nonzero().flatten()
    calib = (~fitting).nonzero().flatten()
    displacement = xyz[:, 5::5] - xyz[:, :-1:5]
    cmd = train[2][..., :3].unflatten(1, (8, 5)).sum(2)
    motion = torch.linalg.lstsq(cmd[fit].reshape(-1, 3), displacement[fit].reshape(-1, 3)).solution
    head.motion.copy_(motion)
    head.inverse_motion.copy_(torch.linalg.pinv(motion))
    head.step_scale.copy_(displacement[fit].flatten(0, 1).std(0).clamp_min(1e-4))
    opt = torch.optim.AdamW(
        [*head.camera.parameters(), *head.trunk.parameters(), *head.waypoints.parameters()], lr=3e-4
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1800, eta_min=3e-5)
    for _ in range(1800):
        ids = fit[torch.randint(len(fit), (128,), device="cuda")]
        _, pred = head.goals(train[0][ids], train[1][ids])
        loss = ((pred - displacement[ids]) / head.step_scale).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        schedule.step()
    with torch.no_grad():
        chunks = [head.experts(train[0][ids], train[1][ids]) for ids in calib.split(128)]
        act, goals, _, context = (torch.cat(v) for v in zip(*chunks, strict=True))
        learned = act[..., :3].unflatten(1, (8, 5)).clamp(-1, 1)
        servo = teacher_servo(head, xyz[calib], goals)
        delta = servo - learned
        residual = train[2][calib, :, :3].unflatten(1, (8, 5)) - learned
        # At deployment only the first block is executed; calibrate against that distribution.
        head.constant_weight.copy_(optimal_weight(delta[:, 0], residual[:, 0]))
        for i in range(10):
            mask = train[3][calib] == i
            head.task_weights[i] = optimal_weight(delta[mask, 0], residual[mask, 0])
    opt = torch.optim.AdamW(head.router.parameters(), lr=3e-4)
    for _ in range(350):
        ids = torch.randint(len(calib), (128,), device="cuda")
        w = head.router(context[ids, 0]).sigmoid()[:, None]
        loss = (residual[ids, 0] - w * delta[ids, 0]).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        test_disp = held_xyz[:, 5::5] - held_xyz[:, :-1:5]
        test_cmd = test[2][..., :3].unflatten(1, (8, 5)).sum(2)
        dynamics = mse_summary(test_disp.flatten(0, 1), (test_cmd @ motion).flatten(0, 1))
        parts = [
            head.experts(test[0][idx], test[1][idx])
            for idx in torch.arange(len(test[1]), device="cuda").split(128)
        ]
        learned, goals, _, ctx = (torch.cat(x) for x in zip(*parts, strict=True))
        learned = learned[:, :5, :3].clamp(-1, 1)
        servo = teacher_servo(head, held_xyz, goals)[:, 0]
        target = test[2][:, :5, :3]
        delta = servo - learned
        oracle = (
            ((target - learned) * delta)
            .sum((1, 2))
            .div(delta.square().sum((1, 2)).clamp_min(1e-8))
            .clamp(0, 1)
        )
        extra = {
            "model_matrix": motion.tolist(),
            "dynamics": dynamics,
            "constant_weight": head.constant_weight.item(),
            "task_weights": head.task_weights.tolist(),
            "test_waypoint_rmse_m": (goals[:, 0] - held_xyz[:, 5]).square().mean().sqrt().item(),
            "oracle_tracking_mse": (learned + oracle[:, None, None] * delta - target).square().mean().item(),
        }
        for mode in head.modes:
            head.mode = mode
            w = head.weights(ctx, test[1])[:, 0]
            tracking = (learned + w[:, None, None] * delta - target).square().mean().item()
            report = {
                "policy_type": "hybrid_control",
                "seed": args.seed,
                "action_offset": 1,
                "task_names": run["task_names"],
                "model": {"expert_config": run["model"], "mode": mode},
                "fitting_windows": len(fit),
                "calibration_windows": len(calib),
                "heldout_windows": len(target),
                "tracking_mse": tracking,
                "test_weight_mean": w.mean().item(),
                "test_weight_std": w.std().item(),
                **extra,
                "per_task": score(head, test, held_records),
            }
            torch.save(head.state_dict(), args.out / f"{mode}-seed{args.seed}.pt")
            (args.out / f"{mode}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
