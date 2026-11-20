"""Train and evaluate a robot action-chunk head using episode-split LIBERO caches."""

import argparse
import json
from pathlib import Path

import torch
from chunk_benchmark import dataset, fit_mask, score

from lerobot.policies.demonstration_chunk.modeling_demonstration_chunk import (
    DUAL_VARIANTS,
    TEMPLATE_VARIANTS,
    VARIANTS,
    DemonstrationChunkHead,
    fit_dictionary,
    motion_coordinates,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tasks", type=int, default=10)
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--reserve-demos", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    train, train_records = dataset(args.cache, "train", args.tasks)
    test, records = dataset(args.cache, "test", args.tasks)
    names = [r["task"].removesuffix("_demo") for r in records]
    offsets = {r.get("action_offset", 0) for r in train_records + records}
    if offsets != {1}:
        raise ValueError(f"Expected next-action labels throughout, got offsets {offsets}")
    for train_record, test_record in zip(train_records, records, strict=True):
        if train_record["task"] != test_record["task"] or set(train_record["demos"]) & set(
            test_record["demos"]
        ):
            raise ValueError("Task alignment or episode split is invalid")
    if args.reserve_demos:
        fitting = fit_mask(train, train_records, args.reserve_demos)
        train = tuple(x[fitting] for x in train)
    config = {"variant": args.variant, "n_tasks": args.tasks}
    if args.variant == "separate_prefix":
        config["width"] = 312
    model = DemonstrationChunkHead(**config).cuda()
    model.state_mean.copy_(train[1][:, :8].mean(0))
    model.state_std.copy_(train[1][:, :8].std(0).clamp_min(1e-3))
    labels = torch.zeros_like(train[3])
    if args.variant in DUAL_VARIANTS:
        labels = labels[:, None].repeat(1, 2)
    if args.variant in TEMPLATE_VARIANTS:
        for i in range(args.tasks):
            mask = train[3] == i
            representatives, assigned = fit_dictionary(train[2][mask], model.chunks.shape[1])
            model.chunks[i].copy_(representatives)
            if args.variant in DUAL_VARIANTS:
                labels[mask, 0] = assigned
                if args.variant in {"dual_prefix", "separate_prefix"}:
                    assigned = torch.cdist(
                        train[2][mask, :5, :6].flatten(1), representatives[:, :5, :6].flatten(1)
                    ).argmin(1)
                labels[mask, 1] = assigned
            else:
                labels[mask] = assigned
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=3e-5)
    for _ in range(args.steps):
        ids = torch.randint(len(labels), (args.batch,), device="cuda")
        loss = model.objective(train[0][ids], train[1][ids], train[2][ids], labels[ids])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        scheduler.step()
    with torch.inference_mode():
        per_task = score(model, test, records)
        for i, row in enumerate(per_task):
            if args.variant not in TEMPLATE_VARIANTS:
                continue
            ids = (test[3] == i).nonzero().flatten()
            target = test[2][ids]
            bank = model.chunks[i]
            oracle = torch.cdist(motion_coordinates(target), motion_coordinates(bank)).argmin(1)
            all_scores = [model(test[0][ix], test[1][ix])[1] for ix in ids.split(128)]
            if args.variant in DUAL_VARIANTS:
                prefix_labels = torch.cdist(target[:, :5, :6].flatten(1), bank[:, :5, :6].flatten(1)).argmin(
                    1
                )
                arm_choice = torch.cat([s[1] for s in all_scores]).argmax(1)
                row["prefix_template_accuracy"] = (arm_choice == prefix_labels).float().mean().item()
                all_scores = [s[0] for s in all_scores]
            chosen = torch.cat(all_scores).argmax(1)
            row["nearest_template_accuracy"] = (chosen == oracle).float().mean().item()
            for name, choice in (("nearest", oracle), ("selected", chosen)):
                row[name + "_template_arm_first5_mse"] = (
                    (bank[choice, :5, :6] - target[:, :5, :6]).square().mean().item()
                )
    report = {
        "policy_type": "demonstration_chunk",
        "task_names": names,
        "model": config,
        "seed": args.seed,
        "action_offset": 1,
        "reserve_demos": args.reserve_demos,
        "steps": args.steps,
        "batch": args.batch,
        "train_windows": len(labels),
        "test_windows": len(test[3]),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "per_task": per_task,
    }
    torch.save(model.state_dict(), args.out / f"{args.variant}-seed{args.seed}.pt")
    (args.out / f"{args.variant}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
