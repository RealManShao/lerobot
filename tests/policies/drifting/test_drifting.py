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

from lerobot.policies.drifting.configuration_drifting import DriftingConfig
from lerobot.policies.drifting.modeling_drifting import DriftingActionHead, DriftingPolicy
from lerobot.policies.factory import get_policy_class, make_policy_config


def _tiny_config() -> DriftingConfig:
    return DriftingConfig(
        device="cpu",
        chunk_size=2,
        n_action_steps=2,
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


def _inputs(config: DriftingConfig, batch_size: int = 3):
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


def test_drifting_policy_is_registered() -> None:
    config = make_policy_config("drifting", device="cpu")
    assert isinstance(config, DriftingConfig)
    assert get_policy_class("drifting").config_class is DriftingConfig


def test_drifting_rejects_multistep_inference() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        DriftingConfig(device="cpu", num_inference_timesteps=4)


def test_raw_base_loader_constructs_drifting_config(tmp_path, monkeypatch) -> None:
    initialized_configs = []

    def fake_init(self, config, **_kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        initialized_configs.append(config)

    monkeypatch.setattr(DriftingPolicy, "__init__", fake_init)

    policy = DriftingPolicy.from_pretrained(tmp_path)

    assert policy.config is initialized_configs[0]
    assert isinstance(policy.config, DriftingConfig)
    assert policy.config.base_model_path == str(tmp_path)


def test_single_sample_geometry_excess_reduces_to_zero() -> None:
    config = _tiny_config()
    head = DriftingActionHead(config)
    geometry = head.compute_geometry_excess(
        observation_embeddings=torch.randn(1, config.action_model_dim),
        actions=torch.randn(1, config.chunk_size, config.max_action_dim),
        action_mask=torch.ones(1, config.chunk_size, config.max_action_dim),
    )
    torch.testing.assert_close(geometry, torch.zeros_like(geometry))


def test_geometry_excess_emphasizes_locally_constrained_coordinate() -> None:
    config = _tiny_config()
    head = DriftingActionHead(config)
    observation_embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    actions = torch.tensor(
        [
            [[0.0, 0.0]],
            [[0.0, 1.0]],
            [[10.0, 0.0]],
            [[10.0, 1.0]],
        ]
    )

    geometry = head.compute_geometry_excess(
        observation_embeddings=observation_embeddings,
        actions=actions,
        action_mask=torch.ones_like(actions),
    )

    assert geometry[0, 0, 0] > geometry[0, 0, 1]
    assert geometry[0, 0, 0] > 0


def test_training_uses_geometry_aware_proposal_and_proximal_losses() -> None:
    config = _tiny_config()
    head = DriftingActionHead(config)
    backbone_output, action_input = _inputs(config)

    output = head(backbone_output, action_input)

    assert output["loss"].ndim == 0
    assert output["proposal_loss"].ndim == 0
    assert output["proximal_loss"].ndim == 0
    assert output["geometry_excess"].shape == action_input["action"].shape
    assert torch.all(output["geometry_excess"] >= 0)
    torch.testing.assert_close(
        output["loss"],
        output["proposal_loss"] + config.proximal_loss_weight * output["proximal_loss"],
    )
    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in head.parameters() if parameter.requires_grad)


def test_inference_is_deterministic_and_uses_one_generator_evaluation() -> None:
    config = _tiny_config()
    head = DriftingActionHead(config).eval()
    backbone_output, action_input = _inputs(config)
    action_input.pop("action")

    with patch.object(head, "_predict", wraps=head._predict) as predict:
        first = head.get_action(backbone_output, action_input)["action_pred"]
        assert predict.call_count == 1
    second = head.get_action(backbone_output, action_input)["action_pred"]

    assert first.shape == (3, config.chunk_size, config.max_action_dim)
    torch.testing.assert_close(first, second)
