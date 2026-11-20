"""Export paired expert proposals at calibration and held-out demonstration states."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from chunk_benchmark import dataset
from train_history_control import samples

from lerobot.policies.demonstration_chunk.hybrid_control import HistoryControlHead, HybridControlHead


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "cache", "out"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 42])
    p.add_argument(
        "--history-suite", type=Path, help="Use causal velocity-corrected goals from this raw suite"
    )
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    models = {}
    for seed in a.seeds:
        root = a.root / f"seed{seed}"
        variant = "history_adaptive" if a.history_suite else "adaptive"
        r = json.loads((root / f"{variant}.json").read_text())
        cls = HistoryControlHead if a.history_suite else HybridControlHead
        m = cls(**r["model"]).cuda().eval()
        m.load_state_dict(torch.load(root / f"{variant}-seed{seed}.pt", weights_only=True))
        models[seed] = m
    for split in ("train", "test"):
        data, records = dataset(a.cache, split, 10)
        if a.history_suite:
            xyz = samples(records, a.history_suite)
            history = torch.stack([xyz[:, 2] - xyz[:, 1], xyz[:, 1] - xyz[:, 0]], 1)
        for i, record in enumerate(records):
            rng = np.random.default_rng(52606 + i)
            rows = []
            ids = []
            for demo in record["demos"][-8:] if split == "train" else record["demos"]:
                number = int(demo.split("_")[-1])
                windows = ((data[3] == i) & (data[4] == number)).nonzero().flatten()
                for section in np.array_split(np.arange(len(windows)), 6):
                    if not len(section):
                        continue
                    index = int(rng.choice(section))
                    idx = windows[index].item()
                    rows.append({"demo": demo, "t": index * 8, "index": idx, "experts": {}})
                    ids.append(idx)
            for seed, model in models.items():
                inputs = [data[0][ids], data[1][ids]]
                if a.history_suite:
                    inputs.append(history[ids])
                plan = model(*inputs)
                arm = plan["action_pred"][:, :5].clone()
                arm[..., :3] = plan["learned_xyz"][:, :5]
                arm.clamp_(-1, 1)
                arm[..., 6] = torch.where(arm[..., 6] > 0, 1.0, -1.0)
                for row, command, goal in zip(rows, arm.tolist(), plan["goals"][:, 0].tolist(), strict=True):
                    row["experts"][seed] = {"command": command, "goal": goal}
            output = {"task": record["task"], "inverse": model.inverse_motion.tolist(), "rows": rows}
            (a.out / f"{i}-{split}.json").write_text(json.dumps(output) + "\n")


if __name__ == "__main__":
    main()
