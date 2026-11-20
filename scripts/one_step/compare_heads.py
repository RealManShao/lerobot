"""Compare single-evaluation DrifOv heads on cached LIBERO observations."""

import argparse
import json
from pathlib import Path

import torch
from cache_libero import HORIZON, TASKS
from chunk_benchmark import dataset, make_batch, score

from lerobot.policies.drif_ov import DrifOvConfig
from lerobot.policies.drif_ov.modeling_drif_ov import DrifOvActionHead


class BlockMeanHead(DrifOvActionHead):
    """Supervise mean translation commands at two time scales on complete suffix blocks."""

    @staticmethod
    def _potential(prediction, target, action_mask, geometry_excess):
        loss = DrifOvActionHead._potential(prediction, target, action_mask, geometry_excess)
        translation_error = (prediction - target)[:, :, :3]
        for size in (5, 20):
            residual = translation_error.unflatten(1, (-1, size)).mean(2)
            valid = action_mask[:, :, :3].unflatten(1, (-1, size)).bool().all(2)
            loss = loss + 0.25 * (residual.square() * valid).sum() / valid.sum().clamp_min(1)
        return loss


class GripperDetailHead(DrifOvActionHead):
    """Extra supervision on gripper sign or direction within the predictable suffix."""

    def _potential(self, prediction, target, action_mask, geometry_excess):
        loss = super()._potential(prediction, target, action_mask, geometry_excess)
        expert = target[:, :, 6]
        grip = prediction[:, :, 6]
        changed = (expert[:, 1:] > 0) != (expert[:, :-1] > 0)
        valid = action_mask[:, 1:, 6].bool() & action_mask[:, :-1, 6].bool()
        if self.use_change_loss:
            # Signed changes distinguish opening from closing. Include stable pairs as negatives.
            residual = (grip.diff(dim=1) - expert.diff(dim=1)).square()
            groups = [(residual, valid & changed), (residual, valid & ~changed)]
        else:
            # Emphasize the two commands straddling each event and their immediate neighbors.
            event_rows = torch.zeros_like(expert, dtype=torch.bool)
            event_rows[:, 1:] |= changed & valid
            event_rows[:, :-1] |= changed & valid
            near = torch.nn.functional.max_pool1d(event_rows[:, None].float(), 3, 1, 1)[:, 0].bool()
            error = torch.nn.functional.binary_cross_entropy_with_logits(
                4 * grip, (expert > 0).float(), reduction="none"
            )
            groups = [(error, near & action_mask[:, :, 6].bool())]
        for values, mask in groups:
            loss = loss + 0.25 * (values * mask).sum() / mask.sum().clamp_min(1)
        return loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "baseline",
            "block_mean",
            "switch_sampling",
            "proximal_1",
            "proximal_0",
            "prox1_blocks",
            "prox1_sign",
            "prox1_change",
        ],
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 1000])
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tf32", action="store_true")
    args = parser.parse_args()
    if args.tf32:
        torch.set_float32_matmul_precision("high")
    train, _ = dataset(args.cache, "train")
    test, records = dataset(args.cache, "test")
    switch = ((train[2][:, 1:, 6] > 0) != (train[2][:, :-1, 6] > 0)).any(1)
    report = {
        "variant": args.variant,
        "steps": args.steps,
        "torch": torch.__version__,
        "train_windows": len(switch),
        "train_event_fraction": switch.float().mean().item(),
        "horizon": HORIZON,
        "batch_size": args.batch_size,
        "layers": args.layers,
        "width": args.width,
        "tf32": args.tf32,
        "results": [],
    }
    # Draw tasks uniformly in every arm. Event oversampling changes only the within-task mixture.
    task_pools = [(train[3] == i).nonzero().flatten() for i in range(len(TASKS))]
    event_pools = [ids[switch[ids]] for ids in task_pools]
    for seed in args.seeds:
        torch.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        cfg = DrifOvConfig(
            device="cuda",
            chunk_size=HORIZON,
            n_action_steps=HORIZON,
            max_state_dim=train[1].shape[-1],
            max_action_dim=7,
            max_num_embodiments=1,
            backbone_embedding_dim=train[0].shape[-1],
            action_model_dim=args.width,
            action_model_num_layers=args.layers,
            action_model_num_heads=4,
            action_model_ff_dim=2 * args.width,
            use_bf16=False,
            proximal_loss_weight=1.0 if args.variant.startswith(("prox1_", "proximal_")) else 81.0,
            use_proximal_loss=args.variant != "proximal_0",
        )
        head_cls = DrifOvActionHead
        if args.variant in ("block_mean", "prox1_blocks"):
            head_cls = BlockMeanHead
        elif args.variant in ("prox1_sign", "prox1_change"):
            head_cls = GripperDetailHead
        head = head_cls(cfg).cuda().train()
        if isinstance(head, GripperDetailHead):
            head.use_change_loss = args.variant == "prox1_change"
        optim = torch.optim.AdamW(head.parameters(), lr=3e-4)
        for _ in range(args.steps):
            tasks = torch.randint(len(TASKS), (args.batch_size,), generator=generator)
            quantiles = torch.rand(args.batch_size, generator=generator)
            oversample = torch.rand(args.batch_size, generator=generator) < 0.5
            selected = torch.empty(args.batch_size, dtype=torch.long, device="cuda")
            for i, pool in enumerate(task_pools):
                rows = tasks == i
                selected[rows] = pool[(quantiles[rows] * len(pool)).long()]
                if args.variant == "switch_sampling":
                    event_rows = rows & oversample
                    events = event_pools[i]
                    selected[event_rows] = events[(quantiles[event_rows] * len(events)).long()]
            obs, batch = make_batch(train, selected)
            optim.zero_grad(set_to_none=True)
            loss = head(obs, batch)["loss"]
            loss.backward()
            optim.step()
        report["results"].append({"seed": seed, "tasks": score(head, test, records)})
        torch.save(head.state_dict(), args.out.parent / f"{args.variant}-seed{seed}.pt")
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.out)


if __name__ == "__main__":
    main()
