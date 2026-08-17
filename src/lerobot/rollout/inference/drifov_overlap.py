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

"""Single-thread chunk-overlap inference for drif_ov-style deployment.

Unlike RTC, this backend runs fully inline on the control thread. It re-plans
before the current chunk is exhausted and conditions the next chunk on the
remaining tail via ``prev_chunk_left_over``.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from copy import copy
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline

from .base import InferenceEngine

logger = logging.getLogger(__name__)


class DrifOvOverlapInferenceEngine(InferenceEngine):
    """Inline chunk-overlap inference with prefix-conditioned re-planning.

    The engine keeps an internal action chunk queue. When remaining actions are
    below ``overlap_steps``, it predicts a new chunk conditioned on the unconsumed
    tail of the previous chunk and atomically swaps to the new chunk.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
        overlap_steps: int,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        requested_overlap_steps = max(0, int(overlap_steps))
        configured_steps = max(1, int(getattr(self._policy.config, "n_action_steps", 1)))
        max_effective_overlap = max(0, configured_steps - 1)
        self._overlap_steps = min(requested_overlap_steps, max_effective_overlap)
        if self._overlap_steps != requested_overlap_steps:
            logger.warning(
                "Requested overlap_steps=%d is not valid for n_action_steps=%d; using overlap_steps=%d",
                requested_overlap_steps,
                configured_steps,
                self._overlap_steps,
            )

        self._processed_queue: torch.Tensor | None = None
        self._original_queue: torch.Tensor | None = None
        self._cursor: int = 0

        logger.info(
            "DrifOvOverlapInferenceEngine initialized "
            "(device=%s, overlap_steps=%d, action_keys=%d)",
            self._device,
            self._overlap_steps,
            len(ordered_action_keys),
        )

    def start(self) -> None:
        logger.info("DrifOvOverlapInferenceEngine started (inline mode)")

    def stop(self) -> None:
        logger.info("DrifOvOverlapInferenceEngine stopped")

    def reset(self) -> None:
        logger.info("Resetting drifov-overlap inference state (policy + processors + chunk queue)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._processed_queue = None
        self._original_queue = None
        self._cursor = 0

    def _remaining_steps(self) -> int:
        if self._processed_queue is None:
            return 0
        return max(0, int(self._processed_queue.shape[0]) - self._cursor)

    def _replan_chunk(self, observation: dict, *, use_prefix: bool) -> None:
        preprocessed = self._preprocessor(observation)

        kwargs: dict[str, Any] = {}
        if use_prefix and self._original_queue is not None and self._remaining_steps() > 0:
            left_over = self._original_queue[self._cursor :].clone()
            if left_over.numel() > 0:
                prefix_valid_steps = (
                    min(int(left_over.shape[0]), self._overlap_steps)
                    if self._overlap_steps > 0
                    else 0
                )
                kwargs = {
                    "prev_chunk_left_over": left_over,
                    "prefix_valid_steps": prefix_valid_steps,
                    "inference_delay": 0,
                    "prefix_is_reanchored": True,
                }

        actions = self._policy.predict_action_chunk(preprocessed, **kwargs)
        original = actions.squeeze(0).clone()
        processed = self._postprocessor(actions).squeeze(0)

        self._original_queue = original
        self._processed_queue = processed
        self._cursor = 0

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        if obs_frame is None:
            return None

        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )

            if self._processed_queue is None or self._remaining_steps() == 0:
                self._replan_chunk(observation, use_prefix=False)
            elif self._overlap_steps > 0 and self._remaining_steps() == self._overlap_steps:
                self._replan_chunk(observation, use_prefix=True)

        if self._processed_queue is None or self._remaining_steps() == 0:
            return None

        action_tensor = self._processed_queue[self._cursor].clone().cpu()
        self._cursor += 1

        action_dict = make_robot_action(action_tensor, self._dataset_features)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])
