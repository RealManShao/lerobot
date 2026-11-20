"""Measure the cost of choosing a whole motion template when only its prefix is executed."""

import argparse
import json
from pathlib import Path

import torch

from lerobot.policies.demonstration_chunk.modeling_demonstration_chunk import (
    fit_dictionary,
    motion_coordinates,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    report = []
    for i in range(10):
        train, test = [
            torch.load(args.cache / f"{i}-{split}.pt", weights_only=True) for split in ("train", "test")
        ]
        if train["action_offset"] != 1 or test["action_offset"] != 1:
            raise ValueError("Expected next-action cache")
        row = {"task": train["task"], "counts": {}, "banks": {}}
        for name, record in (("train", train), ("test", test)):
            grip = record["data"][2][..., 6] > 0
            switches = (grip[:, 1:] != grip[:, :-1]).sum(1)
            row["counts"][name] = torch.bincount(switches, minlength=5).tolist()
        actions, held_out = train["data"][2].cuda(), test["data"][2].cuda()
        for count in (8, 24, 64, 128):
            bank, _ = fit_dictionary(actions, count)
            actions_to_match = held_out[:, :5, :6].flatten(1)
            prefix_distance = torch.cdist(actions_to_match, bank[:, :5, :6].flatten(1)).square() / 30
            full = torch.cdist(motion_coordinates(held_out), motion_coordinates(bank)).argmin(1)
            prefix = prefix_distance.argmin(1)
            row["banks"][count] = {
                "full_match_prefix_mse": prefix_distance.gather(1, full[:, None]).mean().item(),
                "prefix_match_prefix_mse": prefix_distance.min(1).values.mean().item(),
                "same_template_fraction": (full == prefix).float().mean().item(),
                "full_match_grip_accuracy": ((bank[full, :, 6] > 0) == (held_out[:, :, 6] > 0))
                .float()
                .mean()
                .item(),
                "prefix_match_grip_accuracy": ((bank[prefix, :, 6] > 0) == (held_out[:, :, 6] > 0))
                .float()
                .mean()
                .item(),
            }
        report.append(row)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
