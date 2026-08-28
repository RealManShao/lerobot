# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Base rollout strategy: autonomous policy execution with no data recording."""

from __future__ import annotations

from collections import deque
import logging
import time

from lerobot.utils.robot_utils import precise_sleep

from ..context import RolloutContext
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)


class BaseStrategy(RolloutStrategy):
    """Autonomous policy rollout with no data recording.

    All actions flow through the ``robot_action_processor`` pipeline
    before reaching the robot.
    """

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise the inference engine."""
        self._init_engine(ctx)
        logger.info("Base strategy ready")

    def run(self, ctx: RolloutContext) -> None:
        """Run the autonomous control loop until shutdown or duration expires."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)
        target_hz = cfg.fps * interpolator.multiplier

        start_time = time.perf_counter()
        engine.resume()
        logger.info("Base strategy control loop started")

        action_ms_window: deque[float] = deque(maxlen=200)
        total_ms_window: deque[float] = deque(maxlen=200)
        loop_counter = 0

        def _percentile(values: list[float], q: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
            return ordered[idx]

        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()

            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            obs = robot.get_observation()
            observation_end = time.perf_counter()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs)
            processing_end = time.perf_counter()

            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
            action_end = time.perf_counter()
            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            loop_counter += 1
            action_ms_window.append((action_end - processing_end) * 1000.0)
            total_ms_window.append(dt * 1000.0)

            if loop_counter % 120 == 0 and total_ms_window:
                action_values = list(action_ms_window)
                total_values = list(total_ms_window)
                logger.info(
                    "Loop latency stats (window=%d, target=%.1fHz): "
                    "action_p50=%.1fms action_p95=%.1fms total_p50=%.1fms total_p95=%.1fms",
                    len(total_values),
                    target_hz,
                    _percentile(action_values, 0.50),
                    _percentile(action_values, 0.95),
                    _percentile(total_values, 0.50),
                    _percentile(total_values, 0.95),
                )

            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning(
                    "Control loop is running slower (%.1f Hz) than the target control rate (%.1f Hz): "
                    "observation=%.1fms, processing=%.1fms, action=%.1fms, telemetry=%.1fms, total=%.1fms. "
                    "Robot control might be unstable.",
                    1 / dt,
                    target_hz,
                    (observation_end - loop_start) * 1000,
                    (processing_end - observation_end) * 1000,
                    (action_end - processing_end) * 1000,
                    (time.perf_counter() - action_end) * 1000,
                    dt * 1000,
                )

    def teardown(self, ctx: RolloutContext) -> None:
        """Disconnect hardware and stop inference."""
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Base strategy teardown complete")
