from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.robots.tron2 import TRON2_ACTIONS, TRON2_JOINTS, Tron2Robot, Tron2RobotConfig

_MODULE = "lerobot.robots.tron2.tron2"


def test_bridge_observation_uses_remote_cameras_and_state():
    controller = MagicMock()
    controller.is_connected.return_value = True
    provider = MagicMock()
    provider.get_obs.return_value = {
        "state": np.arange(len(TRON2_JOINTS), dtype=np.float32),
        "images": {
            "cam_high": np.zeros((480, 640, 3), dtype=np.uint8),
            "cam_left_wrist": np.ones((480, 640, 3), dtype=np.uint8),
            "cam_right_wrist": np.full((480, 640, 3), 2, dtype=np.uint8),
        },
    }
    provider.get_latest_obs.return_value = provider.get_obs.return_value

    with (
        patch(f"{_MODULE}.require_package"),
        patch(f"{_MODULE}.create_motion_controller", return_value=controller) as create_controller,
        patch(f"{_MODULE}.BridgeObservationProvider", return_value=provider),
    ):
        config = Tron2RobotConfig(observation_source="bridge", bridge_host="wss://bridge.test")
        robot = Tron2Robot(config)
        robot.connect(calibrate=False)
        observation = robot.get_observation()

    assert robot.cameras == {}
    assert set(robot.observation_features) == {
        *TRON2_JOINTS,
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    }
    assert observation["head_yaw.pos"] == len(TRON2_JOINTS) - 1
    assert observation["cam_left_wrist"].shape == (480, 640, 3)
    second_observation = robot.get_observation()
    assert second_observation["cam_left_wrist"].shape == (480, 640, 3)
    controller.get_joint_states.assert_not_called()
    sdk_config = create_controller.call_args.args[0]
    assert sdk_config.init_joints == config.init_joints
    assert sdk_config.init_head == config.init_head
    provider.start.assert_called_once()
    provider.get_obs.assert_called_once()
    provider.get_latest_obs.assert_called_once_with(timeout=config.bridge_timeout)

    robot.disconnect()
    provider.stop.assert_called_once()
    controller.disconnect.assert_called_once()


def test_bridge_mode_rejects_placeholder_host():
    with pytest.raises(ValueError, match="real Bridge WebSocket URL"):
        Tron2RobotConfig(observation_source="bridge", bridge_host="wss://BRIDGE_HOST")


def test_send_action_uses_configured_head_position():
    controller = MagicMock()
    controller.is_connected.return_value = True

    with patch(f"{_MODULE}.require_package"):
        robot = Tron2Robot(Tron2RobotConfig(init_head=[0.25, -0.5]))
    robot.controller = controller
    action = dict(zip(TRON2_ACTIONS, np.arange(len(TRON2_ACTIONS)) / 20, strict=True))

    sent_action = robot.send_action(action)

    assert tuple(robot.action_features) == TRON2_ACTIONS
    assert sent_action == action
    controller.set_gripper.assert_called_once_with(35.0, 75.0)
    np.testing.assert_allclose(
        controller.command_joints.call_args.args[0],
        [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.25, -0.5],
    )
