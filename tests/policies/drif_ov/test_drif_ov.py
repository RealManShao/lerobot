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

from unittest.mock import patch

import pytest
import torch

from lerobot.policies.drif_ov.configuration_drif_ov import DrifOvConfig
from lerobot.policies.drif_ov.modeling_drif_ov import (
    ACTION_PREFIX,
    EXECUTION_OFFSET,
    OVERLAP_LENGTH,
    PREFIX_VALID_MASK,
    DrifOvActionHead,
    DrifOvPolicy,
)
from lerobot.policies.drifting.configuration_drifting import DriftingConfig
from lerobot.policies.drifting.modeling_drifting import DriftingActionHead
from lerobot.policies.factory import get_policy_class, make_policy_config


def _tiny_config(**overrides) -> DrifOvConfig:
    values = {
        "device": "cpu",
        "chunk_size": 4,
        "n_action_steps": 4,
        "max_state_dim": 3,
        "max_action_dim": 2,
        "backbone_embedding_dim": 8,
        "action_model_dim": 8,
        "action_model_num_layers": 2,
        "action_model_num_heads": 2,
        "action_model_ff_dim": 16,
        "action_model_dropout": 0.0,
        "max_num_embodiments": 2,
        "state_dropout_prob": 0.0,
        "use_bf16": False,
        "model_params_fp32": True,
        "max_overlap_steps": 3,
        "prefix_corruption_probability": 0.0,
    }
    values.update(overrides)
    return DrifOvConfig(**values)


def _tiny_drifting_config() -> DriftingConfig:
    return DriftingConfig(
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        max_state_dim=3,
        max_action_dim=2,
        backbone_embedding_dim=8,
        action_model_dim=8,
        action_model_num_layers=2,
        action_model_num_heads=2,
        action_model_ff_dim=16,
        action_model_dropout=0.0,
        max_num_embodiments=2,
        state_dropout_prob=0.0,
        use_bf16=False,
        model_params_fp32=True,
    )


def _inputs(config: DrifOvConfig, batch_size: int = 2):
    backbone_output = {
        "backbone_features": torch.randn(batch_size, 5, config.backbone_embedding_dim),
        "backbone_attention_mask": torch.ones(batch_size, 5, dtype=torch.bool),
        "image_mask": torch.tensor([[False, True, True, False, False]]).expand(batch_size, -1),
    }
    action_input = {
        "state": torch.randn(batch_size, 1, config.max_state_dim),
        "action": torch.randn(batch_size, config.chunk_size, config.max_action_dim),
        "action_mask": torch.ones(batch_size, config.chunk_size, config.max_action_dim),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
    }
    return backbone_output, action_input


def test_drif_ov_policy_is_registered() -> None:
    config = make_policy_config("drif_ov", device="cpu")
    assert isinstance(config, DrifOvConfig)
    assert get_policy_class("drif_ov").config_class is DrifOvConfig


def test_inference_preserves_prefix_with_one_generator_evaluation() -> None:
    config = _tiny_config()
    head = DrifOvActionHead(config).eval()
    backbone_output, action_input = _inputs(config)
    action_input.pop("action")
    prefix = torch.randn(2, config.chunk_size, config.max_action_dim)
    prefix_mask = torch.zeros_like(prefix, dtype=torch.bool)
    prefix_mask[:, :2] = True
    action_input.update(
        {
            ACTION_PREFIX: prefix,
            PREFIX_VALID_MASK: prefix_mask,
            OVERLAP_LENGTH: torch.tensor([2, 2]),
            EXECUTION_OFFSET: torch.tensor([1, 1]),
        }
    )

    with patch.object(head, "_predict", wraps=head._predict) as predict:
        actions = head.get_action(backbone_output, action_input)["action_pred"]

    assert predict.call_count == 1
    torch.testing.assert_close(actions[:, :2], prefix[:, :2])


def test_training_masks_prefix_and_reports_boundary_terms() -> None:
    config = _tiny_config()
    head = DrifOvActionHead(config)
    backbone_output, action_input = _inputs(config)
    prefix_mask = torch.zeros_like(action_input["action"], dtype=torch.bool)
    prefix_mask[:, :2] = True
    action_input[ACTION_PREFIX] = action_input["action"].clone()
    action_input[PREFIX_VALID_MASK] = prefix_mask
    action_input[OVERLAP_LENGTH] = torch.tensor([2, 2])
    action_input[EXECUTION_OFFSET] = torch.tensor([0, 1])

    output = head(backbone_output, action_input)

    assert output["loss"].ndim == 0
    assert output["boundary_loss"].ndim == 0
    assert output["suffix_horizon"].item() == 2
    assert output["prefix_error"].item() == 0
    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in head.parameters() if parameter.requires_grad)


def test_disabled_overlap_modules_match_drifting_head() -> None:
    drifting_config = _tiny_drifting_config()
    config = _tiny_config(
        use_prefix_conditioning=False,
        use_prefix_mask_embedding=False,
        use_overlap_length_embedding=False,
        use_execution_offset_embedding=False,
    )
    drifting_head = DriftingActionHead(drifting_config).eval()
    head = DrifOvActionHead(config).eval()
    head.load_state_dict(drifting_head.state_dict())
    backbone_output, action_input = _inputs(config)
    action_input.pop("action")

    expected = drifting_head.get_action(backbone_output, action_input)["action_pred"]
    actual = head.get_action(backbone_output, action_input)["action_pred"]

    torch.testing.assert_close(actual, expected)


def test_runtime_prefix_length_preserves_zero_action_rows() -> None:
    config = _tiny_config()
    policy = DrifOvPolicy.__new__(DrifOvPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    inputs = {"state": torch.zeros(1, 1, config.max_state_dim)}
    zero_prefix = torch.zeros(3, config.max_action_dim)

    prepared = policy._prepare_overlap_inputs(
        inputs,
        prev_chunk_left_over=zero_prefix,
        prefix_valid_steps=2,
        prefix_valid_mask=None,
        inference_delay=1,
        prefix_is_reanchored=None,
    )

    assert prepared[PREFIX_VALID_MASK][:, :2].all()
    assert not prepared[PREFIX_VALID_MASK][:, 2:].any()
    torch.testing.assert_close(prepared[OVERLAP_LENGTH], torch.tensor([2]))


def test_runtime_rejects_prefix_that_expires_during_inference() -> None:
    config = _tiny_config()
    policy = DrifOvPolicy.__new__(DrifOvPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = config

    with pytest.raises(ValueError, match="stale"):
        policy._prepare_overlap_inputs(
            {"state": torch.zeros(1, 1, config.max_state_dim)},
            prev_chunk_left_over=torch.zeros(3, config.max_action_dim),
            prefix_valid_steps=2,
            prefix_valid_mask=None,
            inference_delay=3,
            prefix_is_reanchored=None,
        )
