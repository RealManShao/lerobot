"""Fit a bounded servo weight for each inferred entity and contact state using calibration demos."""

import argparse
import json
from pathlib import Path

import torch
from chunk_benchmark import dataset, fit_mask
from train_visual_geometry import collect, labels, translation_alternatives

from lerobot.policies.demonstration_chunk.visual_geometry import VisualGeometryHead


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "labels", "root"):
        parser.add_argument("--" + k, type=Path, required=True)
    parser.add_argument("--geometry", required=True, choices=["object", "world", "consistent"])
    parser.add_argument("--seed", type=int, required=True)
    a = parser.parse_args()
    run = json.loads((a.root / f"{a.geometry}_contact.json").read_text())
    if a.seed != run["seed"]:
        raise ValueError("Checkpoint seed mismatch")
    model = {**run["model"], "route": "entity_contact"}
    head = VisualGeometryHead(**model).cuda().eval()
    inc = head.load_state_dict(torch.load(a.root / run["weights"], weights_only=True), strict=False)
    if inc.missing_keys != ["entity_w"] or inc.unexpected_keys:
        raise ValueError(inc)
    data, records = dataset(a.cache, "train", 10)
    truth = labels(a.labels, "train", data, records)
    ids = (~fit_mask(data, records, 8)).nonzero().flatten()
    arm, _, goal, _, _, valid, logit, assoc, _, _ = collect(head, data, ids)
    learned, servo = translation_alternatives(head, arm, goal, data[1][ids, :3])
    delta = servo - learned
    target = truth[2][ids] - data[1][ids, :3] - learned
    shares = head.entity_stage(assoc, valid, logit).flatten(1)
    design = (shares[:, None] * delta[..., None]).reshape(-1, 8)
    lam = 16 * delta.square().sum(1).mean()
    prior = head.stage_w.repeat(4)
    gram = design.T @ design + lam * torch.eye(8, device="cuda")
    rhs = design.T @ target.flatten() + lam * prior
    step = 1 / torch.linalg.eigvalsh(gram).max()
    w = prior.clone()
    for _ in range(250):
        w = (w - step * (gram @ w - rhs)).clamp(0, 1)
    head.entity_w.copy_(w.reshape(4, 2))
    variant = f"{a.geometry}_entity_contact"
    weights = f"{variant}-seed{a.seed}.pt"
    torch.save(head.state_dict(), a.root / weights)
    (a.root / f"{variant}.json").write_text(
        json.dumps(
            {**run, "model": model, "weights": weights, "entity_weight": head.entity_w.tolist()}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
