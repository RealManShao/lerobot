"""Cache frozen GR00T vision features and episode-safe raw LIBERO action windows."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import Qwen2VLImageProcessor

from lerobot.policies.demonstration_chunk.observed_control import encode_rgb_batch
from lerobot.policies.groot.modeling_groot import GrootPolicy

TASKS = ("KITCHEN_SCENE3_", "KITCHEN_SCENE4_", "STUDY_SCENE1_")
HORIZON = 40


@torch.inference_mode()
def encode_views(obs, starts, visual, processor):
    views = []
    for camera in ("agentview_rgb", "eye_in_hand_rgb"):
        images = obs[camera][: starts[-1] + 1 : 8]
        tokens = []
        for first in range(0, len(images), 16):
            batch = torch.from_numpy(images[first : first + 16].copy()).permute(0, 3, 1, 2)
            tokens.append(encode_rgb_batch(batch, visual, processor).cpu())
        views.append(torch.cat(tokens))
    return torch.cat(views, dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument(
        "--action-offset",
        type=int,
        choices=[0, 1],
        default=1,
        help="Use 1 for post-action observations in raw LIBERO HDF5, 0 for legacy replication.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    policy = GrootPolicy.from_pretrained(str(args.checkpoint), local_files_only=True)
    visual = policy._groot_model.backbone.visual.to(device="cuda", dtype=torch.bfloat16).eval()
    del policy
    processor = Qwen2VLImageProcessor(
        patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3
    )
    if args.all_tasks:
        paths = sorted(args.suite.glob("*_demo.hdf5"))
        if len(paths) != 10:
            raise ValueError(f"Expected ten LIBERO-10 tasks, found {len(paths)}")
    else:
        paths = []
        for prefix in TASKS:
            (path,) = args.suite.glob(f"{prefix}*.hdf5")
            paths.append(path)
    for task_id in args.task_ids if args.task_ids is not None else range(len(paths)):
        path = paths[task_id]
        with h5py.File(path) as source:
            group = source["data"]
            names = sorted(group, key=lambda name: int(name.split("_")[-1]))
            order = np.random.default_rng(2026).permutation(names).tolist()
            for split, demos in (
                ("train", order[: len(order) * 4 // 5]),
                ("test", order[len(order) * 4 // 5 :]),
            ):
                chunks = []
                for name in demos:
                    episode = group[name]
                    actions = episode["actions"][:]
                    starts = np.arange(0, len(actions) - HORIZON - args.action_offset + 1, 8)
                    if not len(starts):
                        continue
                    obs = episode["obs"]
                    state = np.concatenate(
                        [obs[k][:][starts] for k in ("ee_pos", "ee_ori", "gripper_states")], 1
                    )
                    chunks.append(
                        (
                            encode_views(obs, starts, visual, processor),
                            torch.from_numpy(state).float(),
                            torch.from_numpy(
                                actions[starts[:, None] + args.action_offset + np.arange(HORIZON)]
                            ).float(),
                            torch.full((len(starts),), int(name.split("_")[-1])),
                        )
                    )
                data = [torch.cat(items) for items in zip(*chunks, strict=True)]
                torch.save(
                    {"task": path.stem, "demos": demos, "action_offset": args.action_offset, "data": data},
                    args.out / f"{task_id}-{split}.pt",
                )


if __name__ == "__main__":
    main()
