"""Measure how much five-step translation error a scalar control mixture could remove."""

import argparse
import json
from pathlib import Path

import torch
from chunk_benchmark import dataset
from train_visual_geometry import collect, labels, translation_alternatives

from lerobot.policies.demonstration_chunk.visual_geometry import VisualGeometryHead


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "cache", "labels", "out"):
        p.add_argument("--" + field, type=Path, required=True)
    args = p.parse_args()
    data, records = dataset(args.cache, "test", 10)
    _, _, end, _ = labels(args.labels, "test", data, records)
    ids = torch.arange(len(end), device=end.device)
    current = data[1][:, :3]
    report = {}
    for geometry in ("world", "object", "consistent"):
        rows = {}
        for seed in (1, 42):
            root = args.root / "step20000" / f"seed{seed}"
            manifest = json.loads((root / f"{geometry}_entity_contact.json").read_text())
            head = VisualGeometryHead(**manifest["model"]).cuda().eval()
            head.load_state_dict(torch.load(root / manifest["weights"], weights_only=True))
            arm, _, goal, _, _, valid, logit, assoc, contact, _ = collect(head, data, ids)
            learned, servo = translation_alternatives(head, arm, goal, current)
            delta = servo - learned
            residual = end - current - learned
            energy = delta.square().sum(1)
            projection = (delta * residual).sum(1) / energy.clamp_min(1e-20)
            weights = {
                "constant": head.global_w.expand_as(energy),
                "contact": torch.lerp(head.stage_w[0], head.stage_w[1], contact),
                "entity_contact": (head.entity_stage(assoc, valid, logit) * head.entity_w).sum((1, 2)),
                "hindsight": projection.clamp(0, 1),
            }
            # The closest point on the segment is a per-window lower bound in this dynamics model.
            error = {k: (residual - w[:, None] * delta).square().sum(1) for k, w in weights.items()}
            for k in ("constant", "contact", "entity_contact"):
                if (error["hindsight"] > error[k] + 1e-8).any():
                    raise ValueError("Invalid projection bound")
            rows[seed] = {
                "windows": len(end),
                "rms_mm": {k: 1000 * v.mean().sqrt().item() for k, v in error.items()},
                "expert_disagreement_rms_mm": 1000 * energy.mean().sqrt().item(),
                "interior_optimum_fraction": ((projection > 0) & (projection < 1)).float().mean().item(),
            }
        report[geometry] = rows
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
