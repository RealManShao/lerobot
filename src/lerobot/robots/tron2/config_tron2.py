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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("tron2")
@dataclass
class Tron2RobotConfig(RobotConfig):
    robot_ip: str = "127.0.0.1"
    port: int = 5000
    state_queue_maxlen: int = 7
    polling_rate: float = 200.0
    connection_timeout: float = 5.0
    state_timeout: float = 1.0
    publish_rate: float = 300.0
    control_frequency: float = 30.0
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("state_timeout", "publish_rate", "control_frequency"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
