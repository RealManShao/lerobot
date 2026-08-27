#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recover a failed final Hub upload from an existing LeRobot training checkpoint.

This script reloads the final checkpoint saved by ``lerobot-train`` and replays only
the upload part of the pipeline:

1. load the train config from the checkpoint's ``pretrained_model/`` directory
2. rebuild the model from that checkpoint
3. rebuild the pre/post processors from that checkpoint
4. push the model and processors to the Hub

It intentionally does not resume training.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.rewards import make_reward_model, make_reward_pre_post_processors
from lerobot.utils.constants import CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK, PRETRAINED_MODEL_DIR


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _resolve_pretrained_dir(source: Path) -> Path:
    source = source.expanduser().resolve()

    if source.is_file():
        if source.name != TRAIN_CONFIG_NAME:
            raise ValueError(f"Expected {TRAIN_CONFIG_NAME}, got: {source}")
        return source.parent

    if not source.is_dir():
        raise FileNotFoundError(source)

    if (source / TRAIN_CONFIG_NAME).is_file():
        return source

    if (source / PRETRAINED_MODEL_DIR / TRAIN_CONFIG_NAME).is_file():
        return source / PRETRAINED_MODEL_DIR

    last_pretrained = source / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK / PRETRAINED_MODEL_DIR
    if (last_pretrained / TRAIN_CONFIG_NAME).is_file():
        return last_pretrained

    raise FileNotFoundError(
        "Could not find a LeRobot checkpoint layout under "
        f"{source}. Expected one of:\n"
        f"  - {TRAIN_CONFIG_NAME}\n"
        f"  - {PRETRAINED_MODEL_DIR}/{TRAIN_CONFIG_NAME}\n"
        f"  - {CHECKPOINTS_DIR}/{LAST_CHECKPOINT_LINK}/{PRETRAINED_MODEL_DIR}/{TRAIN_CONFIG_NAME}"
    )


def _select_load_device(requested_device: str | None) -> str:
    if requested_device is not None:
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_dataset_meta(
    cfg: TrainPipelineConfig,
    *,
    dataset_repo_id: str | None,
    dataset_root: Path | None,
) -> LeRobotDatasetMetadata:
    repo_id = dataset_repo_id or cfg.dataset.repo_id
    root = dataset_root if dataset_root is not None else cfg.dataset.root
    return LeRobotDatasetMetadata(repo_id=repo_id, root=root)


def _restore_config_device(model, original_device: str) -> None:
    if hasattr(model, "config"):
        model.config.device = original_device
    base_model = getattr(model, "model", None)
    if base_model is not None and hasattr(base_model, "config"):
        base_model.config.device = original_device


def _recover_policy_upload(
    cfg: TrainPipelineConfig,
    pretrained_dir: Path,
    *,
    repo_id: str,
    private: bool | None,
    dataset_meta: LeRobotDatasetMetadata,
    load_device: str,
) -> None:
    if cfg.policy is None:
        raise ValueError("Expected a policy config in train_config.json.")

    original_device = str(cfg.policy.device)
    cfg.policy.pretrained_path = pretrained_dir
    cfg.policy.device = load_device
    cfg.policy.repo_id = repo_id
    if private is not None:
        cfg.policy.private = private

    policy = make_policy(cfg.policy, ds_meta=dataset_meta, rename_map=cfg.rename_map)

    cfg.policy.device = original_device
    _restore_config_device(policy, original_device)

    preprocessor_overrides = {
        "device_processor": {"device": load_device},
        "normalizer_processor": {
            "device": load_device,
            "stats": dataset_meta.stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    postprocessor_overrides = {
        "unnormalizer_processor": {
            "device": load_device,
            "stats": dataset_meta.stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        }
    }
    if getattr(cfg.policy, "use_relative_actions", False):
        preprocessor_overrides["relative_actions_processor"] = {
            "enabled": True,
            "exclude_joints": getattr(cfg.policy, "relative_exclude_joints", []),
            "action_names": getattr(cfg.policy, "action_feature_names", None),
        }
        postprocessor_overrides["absolute_actions_processor"] = {"enabled": True}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=pretrained_dir,
        dataset_stats=dataset_meta.stats,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )

    logging.info("Uploading policy to %s", repo_id)
    if cfg.policy.use_peft:
        policy.push_model_to_hub(cfg, peft_model=policy, dataset_meta=dataset_meta)
    else:
        policy.push_model_to_hub(cfg, dataset_meta=dataset_meta)
    preprocessor.push_to_hub(repo_id)
    postprocessor.push_to_hub(repo_id)


def _recover_reward_model_upload(
    cfg: TrainPipelineConfig,
    pretrained_dir: Path,
    *,
    repo_id: str,
    private: bool | None,
    dataset_meta: LeRobotDatasetMetadata,
    load_device: str,
) -> None:
    if cfg.reward_model is None:
        raise ValueError("Expected a reward model config in train_config.json.")

    original_device = str(cfg.reward_model.device)
    cfg.reward_model.pretrained_path = str(pretrained_dir)
    cfg.reward_model.device = load_device
    cfg.reward_model.repo_id = repo_id
    if private is not None:
        cfg.reward_model.private = private

    reward_model = make_reward_model(
        cfg.reward_model,
        dataset_stats=dataset_meta.stats,
        dataset_meta=dataset_meta,
    )

    cfg.reward_model.device = original_device
    _restore_config_device(reward_model, original_device)

    preprocessor, postprocessor = make_reward_pre_post_processors(
        cfg.reward_model,
        dataset_stats=dataset_meta.stats,
        dataset_meta=dataset_meta,
    )

    logging.info("Uploading reward model to %s", repo_id)
    reward_model.push_model_to_hub(cfg)
    preprocessor.push_to_hub(repo_id)
    postprocessor.push_to_hub(repo_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help=(
            "Path to the training output dir, a checkpoint dir, the pretrained_model dir, "
            f"or directly to {TRAIN_CONFIG_NAME}."
        ),
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hub repo to upload to. Defaults to the repo_id saved in the train config.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=None,
        help="Override dataset repo_id when reloading dataset metadata for the model card/processors.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override the local dataset root when reloading dataset metadata.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device used to reload the checkpoint before upload. Defaults to cuda when available, else cpu.",
    )
    parser.add_argument(
        "--private",
        type=_parse_bool,
        default=None,
        help="Override repo privacy for the recovery upload.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    pretrained_dir = _resolve_pretrained_dir(args.source)
    cfg = TrainPipelineConfig.from_pretrained(pretrained_dir)
    dataset_meta = _load_dataset_meta(
        cfg,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
    )

    active_cfg = cfg.trainable_config
    repo_id = args.repo_id or active_cfg.repo_id
    if not repo_id:
        raise ValueError("No repo_id found. Pass --repo-id or ensure the saved train config has one.")

    load_device = _select_load_device(args.device)
    logging.info("Recovered checkpoint: %s", pretrained_dir)
    logging.info("Reload device: %s", load_device)
    logging.info("Upload target: %s", repo_id)

    if cfg.is_reward_model_training:
        _recover_reward_model_upload(
            cfg,
            pretrained_dir,
            repo_id=repo_id,
            private=args.private,
            dataset_meta=dataset_meta,
            load_device=load_device,
        )
    else:
        _recover_policy_upload(
            cfg,
            pretrained_dir,
            repo_id=repo_id,
            private=args.private,
            dataset_meta=dataset_meta,
            load_device=load_device,
        )


if __name__ == "__main__":
    main()
