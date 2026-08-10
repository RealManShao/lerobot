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
from lerobot.policies.drifting.configuration_drifting import DriftingConfig


@PreTrainedConfig.register_subclass("drif_ov")
@dataclass
class DrifOvConfig(DriftingConfig):
    """Overlap-conditioned one-step Drifting policy configuration."""

    use_prefix_conditioning: bool = True
    min_overlap_steps: int = 0
    max_overlap_steps: int | None = None
    zero_overlap_probability: float = 0.2

    prefix_corruption_probability: float = 0.5
    prefix_corruption_std: float = 0.02
    prefix_corruption_clip: float = 0.05

    use_prefix_mask_embedding: bool = True
    use_overlap_length_embedding: bool = True
    use_execution_offset_embedding: bool = True
    max_execution_offset_steps: int | None = None

    use_proximal_loss: bool = True
    use_geometry_weighting: bool = True
    geometry_on_suffix_only: bool = True

    boundary_action_loss_weight: float = 1.0
    boundary_velocity_loss_weight: float = 0.1
    boundary_acceleration_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()

        max_overlap_steps = self.resolved_max_overlap_steps
        if self.min_overlap_steps < 0:
            raise ValueError("min_overlap_steps must be non-negative.")
        if self.min_overlap_steps > max_overlap_steps:
            raise ValueError("min_overlap_steps cannot exceed max_overlap_steps.")
        if not 0.0 <= self.zero_overlap_probability <= 1.0:
            raise ValueError("zero_overlap_probability must be in [0, 1].")
        if not 0.0 <= self.prefix_corruption_probability <= 1.0:
            raise ValueError("prefix_corruption_probability must be in [0, 1].")
        if self.prefix_corruption_std < 0:
            raise ValueError("prefix_corruption_std must be non-negative.")
        if self.prefix_corruption_clip < 0:
            raise ValueError("prefix_corruption_clip must be non-negative.")
        if (
            self.prefix_corruption_probability > 0
            and self.prefix_corruption_std > 0
            and self.prefix_corruption_clip == 0
        ):
            raise ValueError("prefix_corruption_clip must be positive when prefix corruption is enabled.")
        if self.resolved_max_execution_offset_steps < 0:
            raise ValueError("max_execution_offset_steps must be non-negative.")
        if any(
            weight < 0
            for weight in (
                self.boundary_action_loss_weight,
                self.boundary_velocity_loss_weight,
                self.boundary_acceleration_loss_weight,
            )
        ):
            raise ValueError("Boundary loss weights must be non-negative.")

    @property
    def resolved_max_overlap_steps(self) -> int:
        maximum = self.chunk_size - 1 if self.max_overlap_steps is None else self.max_overlap_steps
        if maximum < 0 or maximum >= self.chunk_size:
            raise ValueError("max_overlap_steps must be in [0, chunk_size).")
        return maximum

    @property
    def resolved_max_execution_offset_steps(self) -> int:
        maximum = self.chunk_size if self.max_execution_offset_steps is None else self.max_execution_offset_steps
        if maximum > self.chunk_size:
            raise ValueError("max_execution_offset_steps cannot exceed chunk_size.")
        return maximum
