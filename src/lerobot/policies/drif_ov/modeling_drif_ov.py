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

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.policies.drifting.modeling_drifting import DriftingActionHead, DriftingPolicy
from lerobot.policies.groot.groot_n1_7 import Qwen3Backbone, _tie_unused_qwen_lm_head
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.constants import ACTION

from .configuration_drif_ov import DrifOvConfig

ACTION_PREFIX = "action_prefix"
PREFIX_VALID_MASK = "prefix_valid_mask"
OVERLAP_LENGTH = "overlap_length"
EXECUTION_OFFSET = "execution_offset"


class DrifOvActionHead(DriftingActionHead):
    """One-step Drifting head conditioned on an already planned action prefix."""

    config: DrifOvConfig

    def __init__(self, config: DrifOvConfig) -> None:
        super().__init__(config)
        if config.use_prefix_mask_embedding:
            self.prefix_mask_projection = nn.Linear(self.action_dim, self.model_dim, bias=False)
        if config.use_overlap_length_embedding:
            self.overlap_length_embedding = nn.Embedding(self.action_horizon + 1, self.model_dim)
        if config.use_execution_offset_embedding:
            self.execution_offset_embedding = nn.Embedding(
                config.resolved_max_execution_offset_steps + 1,
                self.model_dim,
            )
        self.set_trainable_parameters(
            tune_projector=config.tune_projector,
            tune_action_model=config.tune_action_model,
            tune_vlln=config.tune_vlln,
        )

    def set_trainable_parameters(
        self,
        tune_projector: bool,
        tune_action_model: bool,
        tune_vlln: bool,
    ) -> None:
        super().set_trainable_parameters(tune_projector, tune_action_model, tune_vlln)
        for name in (
            "prefix_mask_projection",
            "overlap_length_embedding",
            "execution_offset_embedding",
        ):
            module = getattr(self, name, None)
            if module is not None:
                module.requires_grad_(tune_projector)

    def set_frozen_modules_to_eval_mode(self) -> None:
        super().set_frozen_modules_to_eval_mode()
        if self.training and not self.tune_projector:
            for name in (
                "prefix_mask_projection",
                "overlap_length_embedding",
                "execution_offset_embedding",
            ):
                module = getattr(self, name, None)
                if module is not None:
                    module.eval()

    @staticmethod
    def _validate_vector(name: str, value: Tensor, batch_size: int) -> Tensor:
        if value.ndim == 0:
            value = value.expand(batch_size)
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape (B,).")
        return value.long()

    def _validate_prefix(
        self,
        action_prefix: Tensor,
        prefix_valid_mask: Tensor,
        *,
        expected_shape: torch.Size,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if action_prefix.shape != expected_shape:
            raise ValueError(
                f"{ACTION_PREFIX} must have shape {tuple(expected_shape)}, "
                f"got {tuple(action_prefix.shape)}."
            )
        try:
            prefix_valid_mask = prefix_valid_mask.to(dtype=action_prefix.dtype).expand_as(action_prefix)
        except RuntimeError as error:
            raise ValueError(f"{PREFIX_VALID_MASK} must be broadcastable to the action shape.") from error

        prefix_valid_mask = prefix_valid_mask > 0
        prefix_rows = prefix_valid_mask.any(dim=-1)
        overlap_length = prefix_rows.sum(dim=1)
        expected_rows = (
            torch.arange(action_prefix.shape[1], device=action_prefix.device).unsqueeze(0)
            < overlap_length.unsqueeze(1)
        )
        if not torch.equal(prefix_rows, expected_rows):
            raise ValueError("prefix_valid_mask must describe a contiguous prefix starting at timestep zero.")
        return action_prefix, prefix_valid_mask, overlap_length

    def _sample_training_prefix(
        self,
        actions: Tensor,
        action_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, horizon, _ = actions.shape
        valid_steps = action_mask.bool().any(dim=-1).sum(dim=1)
        max_length = torch.minimum(
            (valid_steps - 1).clamp_min(0),
            torch.full_like(valid_steps, self.config.resolved_max_overlap_steps),
        )
        positive_min_length = torch.minimum(
            max_length,
            torch.full_like(max_length, max(1, self.config.min_overlap_steps)),
        )
        sample = torch.rand(batch_size, device=actions.device)
        overlap_length = positive_min_length + torch.floor(
            sample * (max_length - positive_min_length + 1)
        ).long()
        overlap_length = torch.where(max_length > 0, overlap_length, torch.zeros_like(overlap_length))
        if self.config.zero_overlap_probability > 0:
            zero_prefix = (
                torch.rand(batch_size, device=actions.device) < self.config.zero_overlap_probability
            )
            overlap_length = torch.where(zero_prefix, torch.zeros_like(overlap_length), overlap_length)

        prefix_rows = (
            torch.arange(horizon, device=actions.device).unsqueeze(0) < overlap_length.unsqueeze(1)
        )
        prefix_valid_mask = prefix_rows.unsqueeze(-1) & action_mask.bool()
        action_prefix = actions.detach().clone()

        corrupted = torch.zeros(batch_size, dtype=torch.bool, device=actions.device)
        if self.config.prefix_corruption_probability > 0 and self.config.prefix_corruption_std > 0:
            corrupted = (
                torch.rand(batch_size, device=actions.device)
                < self.config.prefix_corruption_probability
            ) & (overlap_length > 0)
            noise = torch.randn_like(action_prefix) * self.config.prefix_corruption_std
            noise = noise.clamp(
                min=-self.config.prefix_corruption_clip,
                max=self.config.prefix_corruption_clip,
            )
            corruption_mask = prefix_valid_mask & corrupted[:, None, None]
            action_prefix = torch.where(corruption_mask, action_prefix + noise, action_prefix)

        max_execution_offset = overlap_length.clamp(
            max=self.config.resolved_max_execution_offset_steps
        )
        execution_offset = torch.floor(
            torch.rand(batch_size, device=actions.device) * (max_execution_offset + 1)
        ).long()
        return action_prefix, prefix_valid_mask, overlap_length, execution_offset, corrupted

    def _resolve_training_prefix(
        self,
        action_input: dict[str, Tensor],
        actions: Tensor,
        action_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not self.config.use_prefix_conditioning:
            batch_size = actions.shape[0]
            return (
                torch.zeros_like(actions),
                torch.zeros_like(actions, dtype=torch.bool),
                torch.zeros(batch_size, device=actions.device, dtype=torch.long),
                torch.zeros(batch_size, device=actions.device, dtype=torch.long),
                torch.zeros(batch_size, device=actions.device, dtype=torch.bool),
            )

        has_prefix = ACTION_PREFIX in action_input
        has_mask = PREFIX_VALID_MASK in action_input
        if has_prefix != has_mask:
            raise ValueError(f"{ACTION_PREFIX} and {PREFIX_VALID_MASK} must be provided together.")
        if not has_prefix:
            return self._sample_training_prefix(actions, action_mask)

        action_prefix, prefix_valid_mask, derived_length = self._validate_prefix(
            action_input[ACTION_PREFIX],
            action_input[PREFIX_VALID_MASK],
            expected_shape=actions.shape,
        )
        if (prefix_valid_mask & ~action_mask.bool()).any():
            raise ValueError("prefix_valid_mask cannot mark padded or otherwise invalid action coordinates.")
        overlap_length = self._validate_vector(
            OVERLAP_LENGTH,
            action_input.get(OVERLAP_LENGTH, derived_length),
            actions.shape[0],
        )
        if not torch.equal(overlap_length, derived_length):
            raise ValueError("overlap_length does not match prefix_valid_mask.")
        if (overlap_length > self.config.resolved_max_overlap_steps).any():
            raise ValueError("overlap_length cannot exceed the configured max_overlap_steps.")
        execution_offset = self._validate_vector(
            EXECUTION_OFFSET,
            action_input.get(EXECUTION_OFFSET, torch.zeros_like(overlap_length)),
            actions.shape[0],
        )
        self._validate_execution_offset(execution_offset)
        if (execution_offset > overlap_length).any():
            raise ValueError("execution_offset cannot exceed overlap_length during training.")
        corrupted = torch.zeros(actions.shape[0], device=actions.device, dtype=torch.bool)
        return action_prefix.detach(), prefix_valid_mask, overlap_length, execution_offset, corrupted

    def _resolve_inference_prefix(
        self,
        action_input: dict[str, Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        empty_prefix = torch.zeros(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
        )
        empty_mask = torch.zeros_like(empty_prefix, dtype=torch.bool)
        empty_length = torch.zeros(batch_size, device=device, dtype=torch.long)
        if not self.config.use_prefix_conditioning or ACTION_PREFIX not in action_input:
            return empty_prefix, empty_mask, empty_length, empty_length
        if PREFIX_VALID_MASK not in action_input:
            raise ValueError(f"{PREFIX_VALID_MASK} is required when {ACTION_PREFIX} is provided.")

        action_prefix, prefix_valid_mask, derived_length = self._validate_prefix(
            action_input[ACTION_PREFIX].to(device=device, dtype=dtype),
            action_input[PREFIX_VALID_MASK].to(device=device),
            expected_shape=empty_prefix.shape,
        )
        overlap_length = self._validate_vector(
            OVERLAP_LENGTH,
            torch.as_tensor(
                action_input.get(OVERLAP_LENGTH, derived_length),
                device=device,
            ),
            batch_size,
        )
        if not torch.equal(overlap_length, derived_length):
            raise ValueError("overlap_length does not match prefix_valid_mask.")
        if (overlap_length > self.config.resolved_max_overlap_steps).any():
            raise ValueError("overlap_length cannot exceed the configured max_overlap_steps.")
        execution_offset = self._validate_vector(
            EXECUTION_OFFSET,
            torch.as_tensor(
                action_input.get(EXECUTION_OFFSET, torch.zeros_like(overlap_length)),
                device=device,
            ),
            batch_size,
        )
        self._validate_execution_offset(execution_offset)
        if (execution_offset > overlap_length).any():
            raise ValueError("The overlap-conditioned request is stale: execution_offset exceeds overlap_length.")
        return action_prefix, prefix_valid_mask, overlap_length, execution_offset

    def _validate_execution_offset(self, execution_offset: Tensor) -> None:
        maximum = self.config.resolved_max_execution_offset_steps
        if (execution_offset < 0).any() or (execution_offset > maximum).any():
            raise ValueError(f"execution_offset must be in [0, {maximum}].")

    def _predict(
        self,
        *,
        vl_features: Tensor,
        state_features: Tensor,
        action_seed: Tensor,
        time: Tensor,
        embodiment_id: Tensor,
        backbone_attention_mask: Tensor,
        image_mask: Tensor,
        prefix_valid_mask: Tensor | None = None,
        overlap_length: Tensor | None = None,
        execution_offset: Tensor | None = None,
    ) -> Tensor:
        batch_size = action_seed.shape[0]
        if prefix_valid_mask is None:
            prefix_valid_mask = torch.zeros_like(action_seed, dtype=torch.bool)
        if overlap_length is None:
            overlap_length = torch.zeros(batch_size, device=action_seed.device, dtype=torch.long)
        if execution_offset is None:
            execution_offset = torch.zeros(batch_size, device=action_seed.device, dtype=torch.long)
        self._validate_execution_offset(execution_offset)

        discrete_time = (time * self.config.num_timestep_buckets).long()
        action_features = self.action_encoder(action_seed, discrete_time, embodiment_id)
        if self.config.use_prefix_mask_embedding:
            action_features = action_features + self.prefix_mask_projection(
                prefix_valid_mask.to(dtype=action_features.dtype)
            )
        if self.config.use_overlap_length_embedding:
            action_features = action_features + self.overlap_length_embedding(overlap_length)[:, None, :]
        if self.config.use_execution_offset_embedding:
            action_features = action_features + self.execution_offset_embedding(execution_offset)[:, None, :]
        if self.config.add_pos_embed:
            position_ids = torch.arange(action_features.shape[1], device=action_features.device)
            action_features = action_features + self.position_embedding(position_ids).unsqueeze(0)

        hidden_states = torch.cat((state_features, action_features), dim=1)
        hidden_states = self.model(
            hidden_states=hidden_states,
            memory=vl_features,
            backbone_attention_mask=backbone_attention_mask,
            image_mask=image_mask,
        )
        decoded = self.action_decoder(hidden_states, embodiment_id)
        return decoded[:, -action_seed.shape[1] :]

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        mask = mask.to(dtype=values.dtype)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def _compute_boundary_terms(
        self,
        prediction: Tensor,
        target: Tensor,
        action_prefix: Tensor,
        action_mask: Tensor,
        overlap_length: Tensor,
    ) -> dict[str, Tensor]:
        batch_size, horizon, _ = target.shape
        batch_indices = torch.arange(batch_size, device=target.device)
        boundary_index = overlap_length.clamp(max=horizon - 1)
        has_boundary = (overlap_length > 0) & (overlap_length < horizon)

        predicted_boundary = prediction[batch_indices, boundary_index]
        target_boundary = target[batch_indices, boundary_index]
        boundary_mask = action_mask[batch_indices, boundary_index].bool() & has_boundary[:, None]
        action_loss = self._masked_mean((predicted_boundary - target_boundary).square(), boundary_mask)

        previous_index = (boundary_index - 1).clamp_min(0)
        prefix_previous = action_prefix[batch_indices, previous_index]
        target_previous = target[batch_indices, previous_index]
        previous_mask = action_mask[batch_indices, previous_index].bool()
        velocity_mask = boundary_mask & previous_mask
        predicted_velocity = predicted_boundary - prefix_previous
        target_velocity = target_boundary - target_previous
        velocity_loss = self._masked_mean(
            (predicted_velocity - target_velocity).square(),
            velocity_mask,
        )

        second_previous_index = (boundary_index - 2).clamp_min(0)
        prefix_second_previous = action_prefix[batch_indices, second_previous_index]
        target_second_previous = target[batch_indices, second_previous_index]
        second_previous_mask = action_mask[batch_indices, second_previous_index].bool()
        acceleration_mask = velocity_mask & second_previous_mask & (overlap_length > 1)[:, None]
        predicted_acceleration = predicted_boundary - 2 * prefix_previous + prefix_second_previous
        target_acceleration = target_boundary - 2 * target_previous + target_second_previous
        acceleration_loss = self._masked_mean(
            (predicted_acceleration - target_acceleration).square(),
            acceleration_mask,
        )

        return {
            "boundary_action_loss": action_loss,
            "boundary_velocity_loss": velocity_loss,
            "boundary_acceleration_loss": acceleration_loss,
            "boundary_action_jump": self._masked_mean(predicted_velocity.abs(), velocity_mask),
            "boundary_velocity_jump": self._masked_mean(
                (predicted_velocity - target_velocity).abs(),
                velocity_mask,
            ),
            "boundary_acceleration_jump": self._masked_mean(
                (predicted_acceleration - target_acceleration).abs(),
                acceleration_mask,
            ),
        }

    def forward(
        self,
        backbone_output: dict[str, Tensor],
        action_input: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        self.set_frozen_modules_to_eval_mode()
        vl_features, state_features, observation_embeddings = self._encode_condition(
            backbone_output,
            action_input,
        )
        actions = action_input["action"]
        action_mask = self._expanded_action_mask(action_input["action_mask"], actions)
        batch_size = actions.shape[0]

        action_prefix, prefix_valid_mask, overlap_length, execution_offset, corrupted = (
            self._resolve_training_prefix(action_input, actions, action_mask)
        )
        prefix_valid_mask = prefix_valid_mask & action_mask.bool()
        suffix_mask = action_mask * (~prefix_valid_mask).to(dtype=action_mask.dtype)
        proposal_seed = torch.where(prefix_valid_mask, action_prefix, torch.zeros_like(actions))
        proposal_raw = self._predict(
            vl_features=vl_features,
            state_features=state_features,
            action_seed=proposal_seed,
            time=torch.zeros(batch_size, device=actions.device, dtype=actions.dtype),
            embodiment_id=action_input["embodiment_id"],
            backbone_attention_mask=backbone_output["backbone_attention_mask"],
            image_mask=backbone_output["image_mask"],
            prefix_valid_mask=prefix_valid_mask,
            overlap_length=overlap_length,
            execution_offset=execution_offset,
        )
        proposal = torch.where(prefix_valid_mask, action_prefix, proposal_raw)

        geometry_mask = suffix_mask if self.config.geometry_on_suffix_only else action_mask
        if self.config.use_geometry_weighting:
            geometry_excess = self.compute_geometry_excess(
                observation_embeddings=observation_embeddings,
                actions=actions,
                action_mask=geometry_mask,
            )
        else:
            geometry_excess = torch.zeros_like(actions)
        proposal_loss = self._potential(proposal, actions, suffix_mask, geometry_excess)

        proximal_loss = torch.zeros((), device=actions.device, dtype=actions.dtype)
        if self.config.use_proximal_loss:
            probe_noise = torch.randn_like(actions) * suffix_mask
            proximal_seed = actions + (1.0 - self.config.proximal_time) * probe_noise
            proximal_seed = torch.where(prefix_valid_mask, action_prefix, proximal_seed)
            proximal_raw = self._predict(
                vl_features=vl_features,
                state_features=state_features,
                action_seed=proximal_seed,
                time=torch.full(
                    (batch_size,),
                    self.config.proximal_time,
                    device=actions.device,
                    dtype=actions.dtype,
                ),
                embodiment_id=action_input["embodiment_id"],
                backbone_attention_mask=backbone_output["backbone_attention_mask"],
                image_mask=backbone_output["image_mask"],
                prefix_valid_mask=prefix_valid_mask,
                overlap_length=overlap_length,
                execution_offset=execution_offset,
            )
            proximal_prediction = torch.where(prefix_valid_mask, action_prefix, proximal_raw)
            proximal_loss = self._potential(
                proximal_prediction,
                actions,
                suffix_mask,
                geometry_excess,
            )

        boundary_terms = self._compute_boundary_terms(
            proposal,
            actions,
            action_prefix,
            action_mask,
            overlap_length,
        )
        boundary_loss = (
            self.config.boundary_action_loss_weight * boundary_terms["boundary_action_loss"]
            + self.config.boundary_velocity_loss_weight * boundary_terms["boundary_velocity_loss"]
            + self.config.boundary_acceleration_loss_weight
            * boundary_terms["boundary_acceleration_loss"]
        )
        loss = proposal_loss + boundary_loss
        if self.config.use_proximal_loss:
            loss = loss + self.config.proximal_loss_weight * proximal_loss

        prefix_error = self._masked_mean((action_prefix - actions).abs(), prefix_valid_mask)
        prefix_preservation_error = self._masked_mean(
            (proposal - action_prefix).abs(),
            prefix_valid_mask,
        )
        suffix_horizon = (~prefix_valid_mask & action_mask.bool()).any(dim=-1).sum(dim=1).float().mean()
        return {
            "loss": loss,
            "proposal_loss": proposal_loss,
            "proximal_loss": proximal_loss,
            "boundary_loss": boundary_loss,
            **boundary_terms,
            "geometry_excess": geometry_excess,
            "overlap_length": overlap_length.float().mean(),
            "execution_offset": execution_offset.float().mean(),
            "prefix_source_expert_fraction": (overlap_length > 0).float().mean(),
            "prefix_corruption_fraction": corrupted.float().mean(),
            "prefix_error": prefix_error,
            "prefix_preservation_error": prefix_preservation_error,
            "suffix_horizon": suffix_horizon,
            "backbone_features": vl_features,
            "state_features": state_features,
        }

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: dict[str, Tensor],
        action_input: dict[str, Tensor],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        if options is not None:
            allowed = {ACTION_PREFIX, PREFIX_VALID_MASK, OVERLAP_LENGTH, EXECUTION_OFFSET}
            unknown = set(options) - allowed
            if unknown:
                raise ValueError(f"Unsupported drif_ov inference options: {sorted(unknown)}.")
            action_input = {**action_input, **options}

        vl_features, state_features, _ = self._encode_condition(backbone_output, action_input)
        batch_size = vl_features.shape[0]
        action_prefix, prefix_valid_mask, overlap_length, execution_offset = (
            self._resolve_inference_prefix(
                action_input,
                batch_size=batch_size,
                device=vl_features.device,
                dtype=vl_features.dtype,
            )
        )
        action_seed = torch.where(prefix_valid_mask, action_prefix, torch.zeros_like(action_prefix))
        prediction = self._predict(
            vl_features=vl_features,
            state_features=state_features,
            action_seed=action_seed,
            time=torch.zeros(batch_size, device=vl_features.device, dtype=vl_features.dtype),
            embodiment_id=action_input["embodiment_id"],
            backbone_attention_mask=backbone_output["backbone_attention_mask"],
            image_mask=backbone_output["image_mask"],
            prefix_valid_mask=prefix_valid_mask,
            overlap_length=overlap_length,
            execution_offset=execution_offset,
        )
        actions = torch.where(prefix_valid_mask, action_prefix, prediction)
        return {
            "action_pred": actions,
            "overlap_length": overlap_length,
            "execution_offset": execution_offset,
            "prefix_valid_mask": prefix_valid_mask,
            "backbone_features": vl_features,
            "state_features": state_features,
        }


class DrifOvN17(nn.Module):
    """Cosmos-Reason2-2B backbone with an overlap-conditioned Drifting head."""

    def __init__(self, config: DrifOvConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = Qwen3Backbone(
            model_name=config.backbone_model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=False,
            use_flash_attention=config.use_flash_attention,
            load_bf16=not config.model_params_fp32,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.model_params_fp32,
            transformers_loading_kwargs={"trust_remote_code": True},
            load_pretrained_weights=True,
        )
        self.action_head = DrifOvActionHead(config)

    def prepare_input(self, inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def move(value: Any) -> Any:
            if not isinstance(value, torch.Tensor):
                return value
            if torch.is_floating_point(value):
                return value.to(self.device, dtype=self.dtype)
            return value.to(self.device)

        return (
            {key: move(value) for key, value in backbone_inputs.items()},
            {key: move(value) for key, value in action_inputs.items()},
        )

    def forward(self, inputs: dict[str, Any]) -> dict[str, Tensor]:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        return self.action_head(self.backbone(backbone_inputs), action_inputs)

    def get_action(
        self,
        inputs: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        _csv = os.environ.get("LEROBOT_PROFILE_INFERENCE_TIMINGS")
        if _csv:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            _t0 = perf_counter()
        backbone_outputs = self.backbone(backbone_inputs)
        if _csv:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            _backbone_s = perf_counter() - _t0
            _t1 = perf_counter()
        result = self.action_head.get_action(backbone_outputs, action_inputs, options)
        if _csv:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            _head_s = perf_counter() - _t1
            _p = Path(_csv)
            _p.parent.mkdir(parents=True, exist_ok=True)
            _line = f"drif_ov,{_backbone_s*1000:.3f},{_head_s*1000:.3f},{(_backbone_s+_head_s)*1000:.3f}\n"
            if not _p.exists():
                _p.write_text("model,backbone_ms,action_head_ms,total_ms\n" + _line)
            else:
                _p.open("a").write(_line)
        return result

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.parameters())).dtype


class DrifOvPolicy(DriftingPolicy):
    name = "drif_ov"
    config_class = DrifOvConfig
    config: DrifOvConfig

    def _create_groot_model(self) -> DrifOvN17:
        model = DrifOvN17(self.config)
        _tie_unused_qwen_lm_head(model.backbone.model)
        if self.config.model_params_fp32:
            self._cast_model_parameters_to_fp32(model)
        return model

    def _filter_groot_inputs(self, batch: dict[str, Tensor], *, include_action: bool) -> dict[str, Tensor]:
        inputs = super()._filter_groot_inputs(batch, include_action=include_action)
        for key in (ACTION_PREFIX, PREFIX_VALID_MASK, OVERLAP_LENGTH, EXECUTION_OFFSET):
            if key in batch:
                inputs[key] = batch[key]
        return inputs

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        inputs = self._filter_groot_inputs(batch, include_action=True)
        device = get_device_from_parameters(self)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=self.config.use_bf16,
        ):
            outputs = self._groot_model.forward(inputs)
        loss = outputs.get("loss")
        if loss is None:
            raise RuntimeError(
                "drif_ov forward did not return a loss. Training batches must include "
                "'action' and 'action_mask'."
            )
        metrics = {
            key: float(value.detach().item())
            for key, value in outputs.items()
            if isinstance(value, Tensor) and value.numel() == 1
        }
        return loss, metrics

    def _prepare_overlap_inputs(
        self,
        inputs: dict[str, Tensor],
        *,
        prev_chunk_left_over: object,
        prefix_valid_steps: object,
        prefix_valid_mask: object,
        inference_delay: object,
        prefix_is_reanchored: object,
    ) -> dict[str, Tensor]:
        if prev_chunk_left_over is None or not self.config.use_prefix_conditioning:
            return inputs
        if not isinstance(prev_chunk_left_over, Tensor):
            raise TypeError("prev_chunk_left_over must be a torch.Tensor for drif_ov.")
        if prev_chunk_left_over.numel() == 0:
            return inputs
        if self.config.use_relative_actions and prefix_is_reanchored is not True:
            raise ValueError(
                "drif_ov requires an explicitly re-anchored prefix when use_relative_actions=true. "
                "Native GR00T relative-action prefixes are not yet supported by the RTC engine."
            )

        prefix = prev_chunk_left_over
        if prefix.ndim == 2:
            prefix = prefix.unsqueeze(0)
        elif prefix.ndim != 3:
            raise ValueError("prev_chunk_left_over must have shape (T, A) or (B, T, A).")

        state = inputs.get("state")
        if state is None:
            raise ValueError("drif_ov overlap conditioning requires state in the preprocessed batch.")
        batch_size = state.shape[0]
        if prefix.shape[0] == 1 and batch_size > 1:
            prefix = prefix.expand(batch_size, -1, -1).clone()
        elif prefix.shape[0] != batch_size:
            raise ValueError("prev_chunk_left_over batch size must match the observation batch size.")

        source_steps = prefix.shape[1]
        if prefix_valid_steps is None:
            valid_steps = torch.full(
                (batch_size,),
                source_steps,
                device=prefix.device,
                dtype=torch.long,
            )
        else:
            valid_steps = torch.as_tensor(prefix_valid_steps, device=prefix.device)
            if valid_steps.ndim == 0:
                valid_steps = valid_steps.expand(batch_size)
            if valid_steps.ndim != 1 or valid_steps.shape[0] != batch_size:
                raise ValueError("prefix_valid_steps must be a scalar or have shape (B,).")
            valid_steps = valid_steps.long()
            if (valid_steps < 0).any() or (valid_steps > source_steps).any():
                raise ValueError("prefix_valid_steps must be within the supplied prefix horizon.")

        prefix = prefix[:, : self.config.chunk_size, : self.config.max_action_dim]
        copied_steps = prefix.shape[1]
        copied_dim = prefix.shape[2]
        padded_prefix = torch.zeros(
            batch_size,
            self.config.chunk_size,
            self.config.max_action_dim,
            device=state.device,
            dtype=state.dtype,
        )
        padded_prefix[:, :copied_steps, :copied_dim] = prefix.to(device=state.device, dtype=state.dtype)

        if prefix_valid_mask is None:
            prefix_rows = (
                torch.arange(self.config.chunk_size, device=state.device).unsqueeze(0)
                < valid_steps.to(device=state.device).clamp(max=copied_steps).unsqueeze(1)
            )
            padded_mask = prefix_rows.unsqueeze(-1).expand(-1, -1, self.config.max_action_dim).clone()
            if copied_dim < self.config.max_action_dim:
                padded_mask[:, :, copied_dim:] = False
        else:
            supplied_mask = torch.as_tensor(prefix_valid_mask, device=state.device)
            if supplied_mask.ndim == 2:
                supplied_mask = supplied_mask.unsqueeze(0)
            if supplied_mask.shape[0] == 1 and batch_size > 1:
                supplied_mask = supplied_mask.expand(batch_size, -1, -1)
            if supplied_mask.ndim != 3 or supplied_mask.shape[0] != batch_size:
                raise ValueError("prefix_valid_mask must have shape (T, A) or (B, T, A).")
            padded_mask = torch.zeros_like(padded_prefix, dtype=torch.bool)
            mask_steps = min(supplied_mask.shape[1], self.config.chunk_size)
            mask_dim = min(supplied_mask.shape[2], self.config.max_action_dim)
            padded_mask[:, :mask_steps, :mask_dim] = supplied_mask[:, :mask_steps, :mask_dim].bool()

        overlap_length = padded_mask.any(dim=-1).sum(dim=1)
        if (overlap_length > self.config.resolved_max_overlap_steps).any():
            raise ValueError(
                "The supplied prefix exceeds max_overlap_steps; align the RTC execution horizon "
                "with the checkpoint training configuration."
            )
        execution_offset = torch.as_tensor(
            0 if inference_delay is None else inference_delay,
            device=state.device,
        )
        if execution_offset.ndim == 0:
            execution_offset = execution_offset.expand(batch_size)
        if execution_offset.ndim != 1 or execution_offset.shape[0] != batch_size:
            raise ValueError("inference_delay must be a scalar or have shape (B,).")
        execution_offset = execution_offset.long()
        maximum_offset = self.config.resolved_max_execution_offset_steps
        if (execution_offset < 0).any() or (execution_offset > maximum_offset).any():
            raise ValueError(f"inference_delay must be in [0, {maximum_offset}] for drif_ov.")
        if (execution_offset > overlap_length).any():
            raise ValueError(
                "The drif_ov request is stale because inference_delay exceeds the valid prefix length."
            )

        overlap_inputs = dict(inputs)
        overlap_inputs[ACTION_PREFIX] = padded_prefix
        overlap_inputs[PREFIX_VALID_MASK] = padded_mask
        overlap_inputs[OVERLAP_LENGTH] = overlap_length
        overlap_inputs[EXECUTION_OFFSET] = execution_offset
        return overlap_inputs

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: object) -> Tensor:
        self.eval()
        inputs = self._filter_groot_inputs(batch, include_action=False)
        inputs = self._prepare_overlap_inputs(
            inputs,
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            prefix_valid_steps=kwargs.get("prefix_valid_steps"),
            prefix_valid_mask=kwargs.get("prefix_valid_mask"),
            inference_delay=kwargs.get("inference_delay"),
            prefix_is_reanchored=kwargs.get("prefix_is_reanchored"),
        )
        device = get_device_from_parameters(self)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=self.config.use_bf16,
        ):
            outputs = self._groot_model.get_action(inputs)

        actions = outputs["action_pred"]
        prediction_horizon = self._resolve_prediction_horizon(actions)
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :prediction_horizon, :original_action_dim]
