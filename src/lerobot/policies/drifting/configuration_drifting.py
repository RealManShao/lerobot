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

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.groot.configuration_groot import GROOT_N1_7_BACKBONE_MODEL, GrootConfig


@PreTrainedConfig.register_subclass("drifting")
@dataclass
class DriftingConfig(GrootConfig):
    """GR00T N1.7 VLM with Implicit Drifting Policy action generation."""

    backbone_model_name: str = GROOT_N1_7_BACKBONE_MODEL
    backbone_embedding_dim: int = 2048
    select_layer: int = 16
    state_history_length: int = 1
    max_num_embodiments: int = 32

    action_model_dim: int = 1024
    action_model_num_layers: int = 12
    action_model_num_heads: int = 16
    action_model_ff_dim: int = 4096
    action_model_dropout: float = 0.1
    attend_text_every_n_blocks: int = 2
    add_pos_embed: bool = True
    max_seq_len: int = 1024
    num_timestep_buckets: int = 1000

    proximal_time: float = 0.9
    proximal_loss_weight: float = 81.0
    geometry_epsilon: float = 1e-6
    state_dropout_prob: float = 0.2

    tune_action_model: bool = True
    tune_diffusion_model: bool = False
    num_inference_timesteps: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backbone_model_name != GROOT_N1_7_BACKBONE_MODEL:
            raise ValueError(
                "Drifting currently supports only the Cosmos-Reason2-2B backbone "
                f"('{GROOT_N1_7_BACKBONE_MODEL}')."
            )
        if self.action_model_dim < 1:
            raise ValueError("action_model_dim must be positive.")
        if self.action_model_num_heads < 1:
            raise ValueError("action_model_num_heads must be at least 1.")
        if self.action_model_dim % self.action_model_num_heads != 0:
            raise ValueError("action_model_dim must be divisible by action_model_num_heads.")
        if self.action_model_num_layers < 1:
            raise ValueError("action_model_num_layers must be at least 1.")
        if self.action_model_ff_dim < 1:
            raise ValueError("action_model_ff_dim must be positive.")
        if not 0.0 <= self.action_model_dropout < 1.0:
            raise ValueError("action_model_dropout must be in [0, 1).")
        if self.attend_text_every_n_blocks < 1:
            raise ValueError("attend_text_every_n_blocks must be at least 1.")
        if self.max_seq_len < self.chunk_size:
            raise ValueError("max_seq_len must be at least chunk_size.")
        if self.num_timestep_buckets < 1:
            raise ValueError("num_timestep_buckets must be positive.")
        if not 0.0 < self.proximal_time < 1.0:
            raise ValueError("proximal_time must be in the open interval (0, 1).")
        if self.proximal_loss_weight < 0:
            raise ValueError("proximal_loss_weight must be non-negative.")
        if self.geometry_epsilon <= 0:
            raise ValueError("geometry_epsilon must be positive.")
        if self.num_inference_timesteps != 1:
            raise ValueError("Drifting uses exactly one inference network evaluation.")
