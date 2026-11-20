"""Fit global, goal-identity, goal-progress and full-observation arbitration rules."""

import argparse
import json
from pathlib import Path

import torch
from chunk_benchmark import dataset
from torch.nn import functional

from lerobot.policies.demonstration_chunk.hybrid_control import OutcomeControlHead


def features(head, cache, root, split, seed):
    data, _ = dataset(cache, split, 10)
    rows, ids, true_task = [], [], []
    for task in range(10):
        part = json.loads((root / f"{task}-{split}.json").read_text())["rows"]
        rows.extend(part)
        ids.extend(r["index"] for r in part)
        true_task.extend([task] * len(part))
    idx = torch.tensor(ids, device="cuda")
    tasks = data[3][idx]
    torch.testing.assert_close(tasks, tasks.new_tensor(true_task))
    with torch.no_grad():
        ctx = torch.cat([head.experts(data[0][b], data[1][b])[3][:, 0] for b in idx.split(128)])
    cost = torch.tensor([r["costs"][str(seed)] for r in rows], device="cuda")
    phase = tasks.new_tensor([r["bits"] for r in rows])
    return ctx, tasks, data[1][idx], phase, cost


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "interventions", "root"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    cfg = json.loads((args.root / "adaptive.json").read_text())
    # LIBERO goal predicates are known from the task specifications; their runtime truth is hidden.
    nbits = max(
        len(json.loads((args.interventions / f"{i}-train.json").read_text())["predicates"]) for i in range(10)
    )
    model = {
        "expert_config": cfg["model"]["expert_config"],
        "mode": "outcome_phase",
        "phase_classes": 2**nbits,
    }
    head = OutcomeControlHead(**model).cuda()
    prior = torch.load(args.root / f"adaptive-seed{args.seed}.pt", weights_only=True)
    incompatible = head.load_state_dict(prior, strict=False)
    if incompatible.unexpected_keys or any(
        not k.startswith(("phase_", "cost_", "global_choice", "task_choice"))
        for k in incompatible.missing_keys
    ):
        raise ValueError(incompatible)
    train = features(head, args.cache, args.interventions, "train", args.seed)
    test = features(head, args.cache, args.interventions, "test", args.seed)
    ctx, tasks, _, phase, parts = train
    # Squared metres at endpoint: equal weight on EEF and mean movable-body marker error.
    cost = parts.sum(-1)
    center = cost - cost.mean(1, keepdim=True)
    scale = center.abs().mean().clamp_min(1e-12)
    global_cost = cost.mean(0)
    head.global_choice.copy_(global_cost.argmin())
    for i in range(10):
        task_mask = tasks == i
        task_mean = (cost[task_mask].sum(0) + 12 * global_cost) / (task_mask.sum() + 12)
        head.task_choice[i] = task_mean.argmin()
        for b in range(2**nbits):
            group = cost[task_mask & (phase == b)]
            head.phase_choice[i, b] = (group.sum(0) + 12 * task_mean).argmin()
    opt = torch.optim.AdamW([*head.phase_model.parameters(), *head.cost_model.parameters()], lr=1e-3)
    for _ in range(400):
        ids = torch.randint(len(ctx), (96,), device="cuda")
        loss = functional.cross_entropy(head.phase_model(ctx[ids]), phase[ids])
        loss = loss + functional.smooth_l1_loss(head.cost_model(ctx[ids]), center[ids] / scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    x, task, state, actual, held = test
    with torch.no_grad():
        phase_hat = head.phase_model(x).argmax(1)
        majority = torch.stack([phase[tasks == i].mode().values for i in range(10)])
        for mode in head.outcome_modes:
            head.mode = mode
            chosen = (head.weights(x[:, None], state)[:, 0] * 2).long()
            select = held[torch.arange(len(x), device=x.device), chosen]
            report = {
                "policy_type": "outcome_control",
                "model": {**model, "mode": mode},
                "task_names": cfg["task_names"],
                "seed": args.seed,
                "action_offset": 1,
                "anchors_train": len(ctx),
                "anchors_test": len(x),
                "cost_train_by_action_m2": cost.mean(0).tolist(),
                "cost_test_by_action_m2": held.sum(-1).mean(0).tolist(),
                "oracle_test_m2": held.sum(-1).min(1).values.mean().item(),
                "selected_test_m2": select.sum(-1).mean().item(),
                "test_selected_eef_m2": select[:, 0].mean().item(),
                "test_selected_object_m2": select[:, 1].mean().item(),
                "phase_accuracy": (phase_hat == actual).float().mean().item(),
                "task_majority_accuracy": (majority[task] == actual).float().mean().item(),
                "test_choices": chosen.bincount(minlength=3).tolist(),
                "task_choices": head.task_choice.tolist(),
                "phase_choices": head.phase_choice.tolist(),
                "global_choice": head.global_choice.item(),
                "test_choices_by_predicate_mask": {
                    str(b): chosen[actual == b].bincount(minlength=3).tolist()
                    for b in actual.unique().tolist()
                },
            }
            torch.save(head.state_dict(), args.root / f"{mode}-seed{args.seed}.pt")
            (args.root / f"{mode}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
