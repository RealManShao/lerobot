"""Keep a trained independent arm fixed and supervise gripper commands directly."""

import argparse
import json
from pathlib import Path

import torch
from chunk_benchmark import dataset, score
from torch.nn import functional

from lerobot.policies.demonstration_chunk.gripper_control import GripperControlHead


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--mode", choices=["grip_binary", "grip_event"], required=True)
    a = p.parse_args()
    r = json.loads((a.root / "separate_prefix.json").read_text())
    if r["seed"] != a.seed or r["action_offset"] != 1 or r.get("reserve_demos", 0) != 0:
        raise ValueError("Expected aligned full-training separate-arm expert")
    torch.manual_seed(a.seed)
    torch.set_float32_matmul_precision("high")
    model = {"expert_config": r["model"], "mode": a.mode}
    head = GripperControlHead(**model).cuda()
    head.expert.load_state_dict(torch.load(a.root / f"separate_prefix-seed{a.seed}.pt", weights_only=True))
    head.expert.requires_grad_(False)
    train, _ = dataset(a.cache, "train", 10)
    test, records = dataset(a.cache, "test", 10)
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1800, eta_min=3e-5)
    for _ in range(1800):
        idx = torch.randint(len(train[0]), (128,), device="cuda")
        pred = head.command(train[0][idx], train[1][idx])
        loss = functional.binary_cross_entropy_with_logits(pred, (train[2][idx, :, 6] > 0).float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    report = {
        "policy_type": "gripper_control",
        "model": model,
        "task_names": r["task_names"],
        "seed": a.seed,
        "action_offset": 1,
        "per_task": score(head, test, records),
        "gripper_trainable_parameters": sum(p.numel() for p in head.parameters() if p.requires_grad),
    }
    torch.save(head.state_dict(), a.root / f"{a.mode}-seed{a.seed}.pt")
    (a.root / f"{a.mode}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
