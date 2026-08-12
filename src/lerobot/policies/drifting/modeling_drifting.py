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
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.groot.groot_n1_7 import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
    Qwen3Backbone,
    _tie_unused_qwen_lm_head,
)
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.constants import ACTION

from .action_head import DriftingTransformer
from .configuration_drifting import DriftingConfig


class DriftingActionHead(nn.Module):
    """Implicit Drifting Policy head with one-step deployment inference."""

    def __init__(self, config: DriftingConfig) -> None:
        super().__init__()
        self.config = config
        self.action_dim = config.max_action_dim
        self.action_horizon = config.chunk_size
        self.model_dim = config.action_model_dim

        self.vlln = nn.LayerNorm(config.backbone_embedding_dim)
        self.vl_projector = nn.Linear(config.backbone_embedding_dim, self.model_dim)
        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.model_dim,
            output_dim=self.model_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.model_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.model = DriftingTransformer(
            dim=self.model_dim,
            num_layers=config.action_model_num_layers,
            num_heads=config.action_model_num_heads,
            ff_dim=config.action_model_ff_dim,
            dropout=config.action_model_dropout,
            attend_text_every_n_blocks=config.attend_text_every_n_blocks,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.model_dim,
            hidden_dim=self.model_dim,
            output_dim=self.action_dim,
        )
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.model_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

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
        self.tune_projector = tune_projector
        self.tune_action_model = tune_action_model
        self.tune_vlln = tune_vlln
        for parameter in self.parameters():
            parameter.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_action_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_projector.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self) -> None:
        if not self.training:
            return
        if not self.tune_projector:
            self.state_encoder.eval()
            self.action_encoder.eval()
            self.action_decoder.eval()
            if self.config.add_pos_embed:
                self.position_embedding.eval()
        if not self.tune_action_model:
            self.model.eval()
        if not self.tune_vlln:
            self.vlln.eval()
            self.vl_projector.eval()

    @staticmethod
    def _expanded_action_mask(action_mask: Tensor, actions: Tensor) -> Tensor:
        if action_mask.ndim == actions.ndim - 1:
            action_mask = action_mask.unsqueeze(-1)
        try:
            return action_mask.to(dtype=actions.dtype).expand_as(actions)
        except RuntimeError as error:
            raise ValueError("action_mask must be broadcastable to the action tensor shape.") from error

    def compute_geometry_excess(
        self,
        observation_embeddings: Tensor,
        actions: Tensor,
        action_mask: Tensor,
    ) -> Tensor:
        """Compute the paper's detached diagonal local geometry excess."""

        eps = self.config.geometry_epsilon
        mask = self._expanded_action_mask(action_mask, actions).detach()
        detached_actions = actions.detach()
        features = F.normalize(observation_embeddings.detach(), dim=-1, eps=eps)

        similarities = features @ features.transpose(0, 1)
        row_mean = similarities.mean(dim=1, keepdim=True)
        row_std = similarities.std(dim=1, correction=0, keepdim=True)
        weights = torch.softmax((similarities - row_mean) / (row_std + eps), dim=1)

        pairwise_differences = detached_actions.unsqueeze(0) - detached_actions.unsqueeze(1)
        weighted_neighbor_mask = weights[:, :, None, None] * mask.unsqueeze(0)
        conditional_variance = (weighted_neighbor_mask * pairwise_differences.square()).sum(
            dim=1
        ) / weighted_neighbor_mask.sum(dim=1).clamp_min(eps)

        coordinate_count = mask.sum(dim=0)
        reference_mean = (detached_actions * mask).sum(dim=0) / coordinate_count.clamp_min(1.0)
        reference_variance = ((detached_actions - reference_mean).square() * mask).sum(
            dim=0
        ) / coordinate_count.clamp_min(1.0)

        conditional_precision = torch.rsqrt(conditional_variance + eps)
        conditional_mean_precision = (conditional_precision * mask).sum(dim=(1, 2), keepdim=True) / mask.sum(
            dim=(1, 2), keepdim=True
        ).clamp_min(1.0)
        conditional_scale = conditional_precision / conditional_mean_precision.clamp_min(eps)

        reference_valid = (coordinate_count > 0).to(dtype=actions.dtype)
        reference_precision = torch.rsqrt(reference_variance + eps)
        reference_mean_precision = (
            reference_precision * reference_valid
        ).sum() / reference_valid.sum().clamp_min(1.0)
        reference_scale = reference_precision / reference_mean_precision.clamp_min(eps)

        geometry_excess = F.relu(conditional_scale / (reference_scale.unsqueeze(0) + eps) - 1.0)
        return (geometry_excess * mask).detach()

    def _encode_condition(
        self,
        backbone_output: dict[str, Tensor],
        action_input: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        vl_features = self.vl_projector(self.vlln(backbone_output["backbone_features"]))
        state = action_input["state"]
        if state.shape[1] != self.config.state_history_length:
            raise ValueError("state history length does not match DriftingConfig.")
        state = state.view(state.shape[0], 1, -1)
        state_features = self.state_encoder(state, action_input["embodiment_id"])

        if self.training and self.config.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.config.state_dropout_prob
            )
            state_features = state_features * (1 - do_dropout[:, None, None].to(dtype=state_features.dtype))

        attention_mask = backbone_output["backbone_attention_mask"].to(dtype=vl_features.dtype)
        pooled_vl = (vl_features * attention_mask.unsqueeze(-1)).sum(dim=1)
        pooled_vl = pooled_vl / attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        observation_embeddings = pooled_vl + state_features.squeeze(1)
        return vl_features, state_features, observation_embeddings

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
    ) -> Tensor:
        discrete_time = (time * self.config.num_timestep_buckets).long()
        action_features = self.action_encoder(action_seed, discrete_time, embodiment_id)
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
    def _potential(
        prediction: Tensor,
        target: Tensor,
        action_mask: Tensor,
        geometry_excess: Tensor,
    ) -> Tensor:
        residual = prediction - target
        weighted_error = 0.5 * (1.0 + geometry_excess) * residual.square() * action_mask
        per_sample = weighted_error.sum(dim=(1, 2)) / action_mask.sum(dim=(1, 2)).clamp_min(1.0)
        return per_sample.mean()

    def forward(
        self,
        backbone_output: dict[str, Tensor],
        action_input: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        self.set_frozen_modules_to_eval_mode()
        vl_features, state_features, observation_embeddings = self._encode_condition(
            backbone_output, action_input
        )
        actions = action_input["action"]
        action_mask = self._expanded_action_mask(action_input["action_mask"], actions)
        batch_size = actions.shape[0]

        proposal = self._predict(
            vl_features=vl_features,
            state_features=state_features,
            action_seed=torch.zeros_like(actions),
            time=torch.zeros(batch_size, device=actions.device, dtype=actions.dtype),
            embodiment_id=action_input["embodiment_id"],
            backbone_attention_mask=backbone_output["backbone_attention_mask"],
            image_mask=backbone_output["image_mask"],
        )
        probe_noise = torch.randn_like(actions) * action_mask
        proximal_seed = actions + (1.0 - self.config.proximal_time) * probe_noise
        proximal_prediction = self._predict(
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
        )

        geometry_excess = self.compute_geometry_excess(
            observation_embeddings=observation_embeddings,
            actions=actions,
            action_mask=action_mask,
        )
        proposal_loss = self._potential(proposal, actions, action_mask, geometry_excess)
        proximal_loss = self._potential(proximal_prediction, actions, action_mask, geometry_excess)
        loss = proposal_loss + self.config.proximal_loss_weight * proximal_loss
        return {
            "loss": loss,
            "proposal_loss": proposal_loss,
            "proximal_loss": proximal_loss,
            "geometry_excess": geometry_excess,
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
            raise NotImplementedError("Native RTC overlap guidance is not supported by Drifting.")
        vl_features, state_features, _ = self._encode_condition(backbone_output, action_input)
        batch_size = vl_features.shape[0]
        actions = self._predict(
            vl_features=vl_features,
            state_features=state_features,
            action_seed=torch.zeros(
                batch_size,
                self.action_horizon,
                self.action_dim,
                device=vl_features.device,
                dtype=vl_features.dtype,
            ),
            time=torch.zeros(batch_size, device=vl_features.device, dtype=vl_features.dtype),
            embodiment_id=action_input["embodiment_id"],
            backbone_attention_mask=backbone_output["backbone_attention_mask"],
            image_mask=backbone_output["image_mask"],
        )
        return {
            "action_pred": actions,
            "backbone_features": vl_features,
            "state_features": state_features,
        }

    def prepare_input(self, batch: dict[str, Any]) -> dict[str, Any]:
        return batch


class DriftingN17(nn.Module):
    """Cosmos-Reason2-2B backbone with an Implicit Drifting action head."""

    def __init__(self, config: DriftingConfig) -> None:
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
        self.action_head = DriftingActionHead(config)

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
            _line = f"drifting,{_backbone_s*1000:.3f},{_head_s*1000:.3f},{(_backbone_s+_head_s)*1000:.3f}\n"
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


class DriftingPolicy(GrootPolicy):
    name = "drifting"
    config_class = DriftingConfig

    def _create_groot_model(self) -> DriftingN17:
        model = DriftingN17(self.config)
        _tie_unused_qwen_lm_head(model.backbone.model)
        if self.config.model_params_fp32:
            self._cast_model_parameters_to_fp32(model)
        return model

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: object) -> Tensor:
        if kwargs.get("prev_chunk_left_over") is not None:
            raise NotImplementedError("Drifting does not support RTC overlap guidance.")
        self.eval()
        inputs = self._filter_groot_inputs(batch, include_action=False)
        device = get_device_from_parameters(self)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.get_action(inputs)

        actions = outputs["action_pred"]
        prediction_horizon = self._resolve_prediction_horizon(actions)
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :prediction_horizon, :original_action_dim]
