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

from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.import_utils import _tron2_env_available, require_package

from ..robot import Robot
from .config_tron2 import Tron2RobotConfig

if TYPE_CHECKING or _tron2_env_available:
    from tron2_env import Tron2Config, create_motion_controller
    from tron2_env.motion import MotionController
else:
    MotionController = Any


TRON2_JOINTS = (
    "left_arm_joint_1.pos",
    "left_arm_joint_2.pos",
    "left_arm_joint_3.pos",
    "left_arm_joint_4.pos",
    "left_arm_joint_5.pos",
    "left_arm_joint_6.pos",
    "left_arm_joint_7.pos",
    "left_gripper.pos",
    "right_arm_joint_1.pos",
    "right_arm_joint_2.pos",
    "right_arm_joint_3.pos",
    "right_arm_joint_4.pos",
    "right_arm_joint_5.pos",
    "right_arm_joint_6.pos",
    "right_arm_joint_7.pos",
    "right_gripper.pos",
    "head_pitch.pos",
    "head_yaw.pos",
)

_LEFT_ARM = slice(0, 7)
_LEFT_GRIPPER = 7
_RIGHT_ARM = slice(8, 15)
_RIGHT_GRIPPER = 15
_HEAD = slice(16, 18)


class Tron2Robot(Robot):
    config_class = Tron2RobotConfig
    name = "tron2"

    def __init__(self, config: Tron2RobotConfig):
        require_package("tron2-env", extra="tron2", import_name="tron2_env")
        super().__init__(config)
        self.config = config
        self.controller: MotionController | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return dict.fromkeys(TRON2_JOINTS, float)

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {
            name: (config.height, config.width, 3) for name, config in self.config.cameras.items()
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.controller is not None
            and self.controller.is_connected()
            and all(camera.is_connected for camera in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} is already connected")

        sdk_config = Tron2Config(
            robot_ip=self.config.robot_ip,
            port=self.config.port,
            state_queue_maxlen=self.config.state_queue_maxlen,
            polling_rate=self.config.polling_rate,
            connection_timeout=self.config.connection_timeout,
        )
        self.controller = create_motion_controller(
            sdk_config,
            publish_rate=self.config.publish_rate,
            eta_default=1.0 / self.config.control_frequency,
        )
        try:
            for camera in self.cameras.values():
                camera.connect()
            self.configure()
        except Exception:
            self.disconnect()
            raise

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> RobotObservation:
        if not self.is_connected or self.controller is None:
            raise ConnectionError(f"{self} is not connected")

        state = np.asarray(
            self.controller.get_joint_states(timeout=self.config.state_timeout)["states"],
            dtype=np.float64,
        )
        if state.shape != (len(TRON2_JOINTS),):
            raise RuntimeError(
                f"Expected {len(TRON2_JOINTS)} TRON2 state values, received shape {state.shape}"
            )

        observation: RobotObservation = dict(zip(TRON2_JOINTS, state, strict=True))
        for name, camera in self.cameras.items():
            observation[name] = camera.read_latest()
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected or self.controller is None:
            raise ConnectionError(f"{self} is not connected")

        missing = set(TRON2_JOINTS).difference(action)
        unexpected = set(action).difference(TRON2_JOINTS)
        if missing or unexpected:
            raise ValueError(
                f"TRON2 action keys do not match action_features; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        target = np.asarray([action[name] for name in TRON2_JOINTS], dtype=np.float64)
        servo_target = np.concatenate((target[_LEFT_ARM], target[_RIGHT_ARM], target[_HEAD]))
        grippers = np.clip(
            target[[_LEFT_GRIPPER, _RIGHT_GRIPPER]] * 100.0,
            0.0,
            100.0,
        )
        self.controller.set_gripper(float(grippers[0]), float(grippers[1]))
        self.controller.command_joints(servo_target)
        return dict(zip(TRON2_JOINTS, target, strict=True))

    def disconnect(self) -> None:
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        if self.controller is not None:
            self.controller.disconnect()
            self.controller = None
