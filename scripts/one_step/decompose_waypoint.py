"""Exact held-out Cartesian error decomposition for an object-relative waypoint."""

import argparse
import json
from pathlib import Path

import torch
from chunk_benchmark import dataset
from train_visual_geometry import collect, labels

from lerobot.policies.demonstration_chunk.visual_geometry import VisualGeometryHead, rotation


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "cache", "labels", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--geometry", choices=["object", "consistent"], default="object")
    a = p.parse_args()
    data, records = dataset(a.cache, "test", 10)
    pose, _, end, valid = labels(a.labels, "test", data, records)
    ids = torch.arange(len(end), device="cuda")
    true_pos = pose[..., :3]
    true_rot = rotation(pose[..., 3:])
    true_local = (true_rot.transpose(-1, -2) @ (end[:, None] - true_pos)[..., None]).squeeze(-1)
    nearest = (true_pos - end[:, None]).square().sum(-1).masked_fill(~valid, torch.inf).argmin(1)
    out = {}
    for step in (1800, 7200, 20000):
        for seed in (1, 42):
            root = a.root / f"step{step}" / f"seed{seed}"
            run = json.loads((root / f"{a.geometry}_contact.json").read_text())
            head = VisualGeometryHead(**run["model"]).cuda().eval()
            head.load_state_dict(torch.load(root / run["weights"], weights_only=True))
            _, _, goal, pos, orient, _, _, association, _, offset = collect(head, data, ids)
            weights = association.softmax(1)
            candidates = pos + (orient @ offset[..., None]).squeeze(-1)
            oracle_candidates = true_pos + (true_rot @ offset[..., None]).squeeze(-1)
            oracle_candidates = torch.where(valid[..., None], oracle_candidates, candidates)
            chosen = candidates[ids, nearest]
            # Sum of these vectors is exactly the deployed endpoint error for every window.
            pieces = torch.stack(
                [
                    goal - chosen,
                    (pos - true_pos)[ids, nearest],
                    ((orient - true_rot) @ true_local[..., None]).squeeze(-1)[ids, nearest],
                    (orient @ (offset - true_local)[..., None]).squeeze(-1)[ids, nearest],
                ],
                1,
            )
            torch.testing.assert_close(pieces.sum(1), goal - end, rtol=0, atol=2e-6)
            gram = torch.einsum("bid,bjd->ij", pieces, pieces) * 1e6 / len(end)

            def rms(pred):
                return 1000 * (pred - end).square().sum(1).mean().sqrt().item()

            out[f"{step}-{seed}"] = {
                "terms": ["association", "translation", "rotation", "local_offset"],
                "gram_mm2": gram.tolist(),
                "component_rms_mm": gram.diag().sqrt().tolist(),
                "endpoint_rms_mm": rms(goal),
                "true_frames_only_rms_mm": rms((weights[..., None] * oracle_candidates).sum(1)),
                "true_anchor_only_rms_mm": rms(chosen),
                "true_frames_and_anchor_rms_mm": rms(oracle_candidates[ids, nearest]),
                "invalid_slot_weight": (weights * (~valid)).sum(1).mean().item(),
            }
    a.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
