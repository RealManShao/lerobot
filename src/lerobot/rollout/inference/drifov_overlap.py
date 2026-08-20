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

"""Async chunk-overlap inference for drif_ov-style deployment.

This backend keeps execution on the main control thread but offloads next-chunk
planning to a background thread. The worker prefetches the next chunk while the
current chunk is still being executed, and conditions that plan on the current
leftover prefix via ``prev_chunk_left_over``.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from copy import copy
from threading import Event, Lock, Thread
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline

from .base import InferenceEngine

logger = logging.getLogger(__name__)


class DrifOvOverlapInferenceEngine(InferenceEngine):
    """Async prefetch chunk-overlap inference with prefix-conditioned re-planning.

    The engine keeps an internal action chunk queue. When remaining actions reach
    ``overlap_steps``, it schedules a background re-plan conditioned on the
    unconsumed tail. At chunk boundary, prefetched actions are swapped in
    atomically.
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
        self._next_chunk: tuple[torch.Tensor, torch.Tensor] | None = None

        self._lock = Lock()
        self._shutdown_event = Event()
        self._job_event = Event()
        self._worker: Thread | None = None
        self._pending_job: dict[str, Any] | None = None
        self._planning = False

        logger.info(
            "DrifOvOverlapInferenceEngine initialized "
            "(device=%s, overlap_steps=%d, action_keys=%d)",
            self._device,
            self._overlap_steps,
            len(ordered_action_keys),
        )

    def start(self) -> None:
        self._shutdown_event.clear()
        self._job_event.clear()
        self._worker = Thread(target=self._planning_loop, daemon=True, name="DrifOvOverlapPlanner")
        self._worker.start()
        logger.info("DrifOvOverlapInferenceEngine started (async prefetch mode)")

    def stop(self) -> None:
        self._shutdown_event.set()
        self._job_event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._worker = None
        logger.info("DrifOvOverlapInferenceEngine stopped")

    def reset(self) -> None:
        logger.info("Resetting drifov-overlap inference state (policy + processors + chunk queue)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        with self._lock:
            self._processed_queue = None
            self._original_queue = None
            self._cursor = 0
            self._next_chunk = None
            self._pending_job = None
            self._planning = False
            self._job_event.clear()

    @staticmethod
    def _remaining_steps(queue: torch.Tensor | None, cursor: int) -> int:
        if queue is None:
            return 0
        return max(0, int(queue.shape[0]) - cursor)

    def _plan_chunk(self, observation: dict, *, use_prefix: bool, left_over: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        preprocessed = self._preprocessor(observation)

        kwargs: dict[str, Any] = {}
        if use_prefix and left_over is not None and left_over.numel() > 0:
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
        return original, processed

    def _submit_job(self, observation: dict, *, use_prefix: bool, left_over: torch.Tensor | None) -> None:
        with self._lock:
            if self._planning or self._next_chunk is not None:
                return
            self._pending_job = {
                "observation": observation,
                "use_prefix": use_prefix,
                "left_over": left_over,
            }
            self._planning = True
            self._job_event.set()

    def _planning_loop(self) -> None:
        while not self._shutdown_event.is_set():
            if not self._job_event.wait(timeout=0.02):
                continue
            self._job_event.clear()
            if self._shutdown_event.is_set():
                break

            with self._lock:
                job = self._pending_job
                self._pending_job = None

            if job is None:
                continue

            try:
                autocast_ctx = (
                    torch.autocast(device_type=self._device.type)
                    if self._device.type == "cuda" and self._policy.config.use_amp
                    else nullcontext()
                )
                with torch.inference_mode(), autocast_ctx:
                    planned = self._plan_chunk(
                        job["observation"],
                        use_prefix=bool(job["use_prefix"]),
                        left_over=job["left_over"],
                    )
                with self._lock:
                    self._next_chunk = planned
            except Exception as exc:  # noqa: BLE001
                logger.error("drifov_overlap async planning failed: %s", exc)
            finally:
                with self._lock:
                    self._planning = False

    def _swap_in_next_chunk_locked(self) -> bool:
        if self._next_chunk is None:
            return False
        self._original_queue, self._processed_queue = self._next_chunk
        self._cursor = 0
        self._next_chunk = None
        return True

    def _maybe_schedule_prefetch(self, observation: dict) -> None:
        with self._lock:
            queue = self._processed_queue
            original = self._original_queue
            cursor = self._cursor
            planning = self._planning
            has_next = self._next_chunk is not None

        remaining = self._remaining_steps(queue, cursor)
        if remaining <= 0 or self._overlap_steps <= 0 or planning or has_next:
            return

        if remaining <= self._overlap_steps and original is not None:
            left_over = original[cursor:].clone()
            obs_copy = copy(observation)
            self._submit_job(obs_copy, use_prefix=True, left_over=left_over)

    def _sync_bootstrap_if_needed(self, observation: dict) -> None:
        with self._lock:
            has_active = self._remaining_steps(self._processed_queue, self._cursor) > 0
            has_next = self._next_chunk is not None

        if has_active:
            return
        if has_next:
            with self._lock:
                self._swap_in_next_chunk_locked()
            return

        # First chunk is synchronous to avoid an empty start window.
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            original, processed = self._plan_chunk(observation, use_prefix=False, left_over=None)
        with self._lock:
            self._original_queue = original
            self._processed_queue = processed
            self._cursor = 0

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        if obs_frame is None:
            return None

        observation = copy(obs_frame)
        observation = prepare_observation_for_inference(
            observation, self._device, self._task, self._robot_type
        )

        self._sync_bootstrap_if_needed(observation)
        self._maybe_schedule_prefetch(observation)

        with self._lock:
            if self._remaining_steps(self._processed_queue, self._cursor) <= 0:
                self._swap_in_next_chunk_locked()

            if self._remaining_steps(self._processed_queue, self._cursor) <= 0:
                return None

            queue = self._processed_queue
            if queue is None:
                return None
            action_tensor = queue[self._cursor].clone().cpu()
            self._cursor += 1

            # If current chunk has just been exhausted, swap immediately if ready.
            if self._remaining_steps(self._processed_queue, self._cursor) <= 0:
                self._swap_in_next_chunk_locked()

        action_dict = make_robot_action(action_tensor, self._dataset_features)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])
