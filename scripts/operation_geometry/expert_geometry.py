"""Estimate DrifOv's diagonal metric from disjoint same-task demonstration minibatches."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from cache_libero import TASKS, encode_views
from probe import H, anchors
from transformers import Qwen2VLImageProcessor

from lerobot.policies.drif_ov import DrifOvConfig
from lerobot.policies.drif_ov.modeling_drif_ov import DrifOvActionHead
from lerobot.policies.groot.modeling_groot import GrootPolicy


def sample(group, name, t, token, head, visual, proc):
    ep = group[name]
    obs = ep["obs"]
    imgs = {k: obs[k][t : t + 1] for k in ("agentview_rgb", "eye_in_hand_rgb")}
    vision = encode_views(imgs, np.array([0]), visual, proc).cuda().float()
    state = np.concatenate([obs[k][t] for k in ("ee_pos", "ee_ori", "gripper_states")] + [token])
    batch = {
        "state": torch.tensor(state, device="cuda").float()[None, None],
        "embodiment_id": torch.zeros(1, dtype=torch.long, device="cuda"),
    }
    valid = torch.ones(vision.shape[:2], dtype=torch.bool, device="cuda")
    _, _, emb = head._encode_condition(
        {"backbone_features": vision, "image_mask": valid, "backbone_attention_mask": valid},
        batch,
    )
    return emb[0], torch.from_numpy(ep["actions"][t : t + 40]).cuda().float()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("suite", "checkpoint", "head", "out"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--shard", type=int, required=True)
    a = p.parse_args()
    pol = GrootPolicy.from_pretrained(str(a.checkpoint), local_files_only=True)
    visual = pol._groot_model.backbone.visual.to(device="cuda", dtype=torch.bfloat16).eval()
    del pol
    proc = Qwen2VLImageProcessor(
        patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3
    )
    cfg = DrifOvConfig(
        device="cuda",
        chunk_size=40,
        n_action_steps=40,
        max_state_dim=11,
        max_action_dim=7,
        max_num_embodiments=1,
        backbone_embedding_dim=2048,
        action_model_dim=256,
        action_model_num_layers=4,
        action_model_num_heads=4,
        action_model_ff_dim=512,
        use_bf16=False,
    )
    head = DrifOvActionHead(cfg).cuda().eval()
    head.load_state_dict(torch.load(a.head, weights_only=True))
    a.out.mkdir(parents=True, exist_ok=True)
    for task_id, path in enumerate(sorted(a.suite.glob("*.hdf5"))):
        if task_id % 4 != a.shard:
            continue
        task = path.stem.removesuffix("_demo")
        token = [float(task.startswith(prefix)) for prefix in TASKS]
        with h5py.File(path) as f:
            grp = f["data"]

            rng = np.random.default_rng(19031 + task_id)
            others = sorted(set(grp) - {"demo_0", "demo_1"})
            banks = []
            for _ in range(4):
                rows = [
                    sample(grp, n, rng.integers(len(grp[n]["actions"]) - 39), token, head, visual, proc)
                    for n in rng.choice(others, 31, replace=False)
                ]
                banks.append(tuple(torch.stack(x) for x in zip(*rows, strict=True)))
            out = {}
            for name in ("demo_0", "demo_1"):
                for phase, t in zip(
                    ("before", "switch", "after"), anchors(grp[name]["actions"][:]), strict=True
                ):
                    emb, actions = sample(grp, name, t, token, head, visual, proc)
                    weights = []
                    for bank_emb, bank_a in banks:
                        acts = torch.cat([actions[None], bank_a])
                        g = head.compute_geometry_excess(
                            torch.cat([emb[None], bank_emb]), acts, torch.ones_like(acts)
                        )
                        weights.append((1 + g[0, :H, :3]).tolist())
                    out[name + "--" + phase] = {"start": t, "weights": weights}
        (a.out / (task + ".json")).write_text(json.dumps({"known_task_id": any(token), "anchors": out}))


if __name__ == "__main__":
    main()
