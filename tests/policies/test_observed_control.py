"""The manipulation controller runs from robot observations alone."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.demonstration_chunk.observed_control import ObservedController


class QuadrantEncoder(torch.nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, pixels, grid_thw):
        return SimpleNamespace(pooler_output=pixels.new_ones(16, 1))


class FixedHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("motion", torch.eye(3))
        self.register_buffer("inverse_motion", torch.eye(3))
        self.calls = 0

    def forward(self, views, state):
        assert views.shape == (1, 8, 1) and state.shape == (1, 8)
        self.calls += 1
        return {
            "action_pred": torch.zeros(1, 40, 7),
            "learned_xyz": torch.zeros(1, 40, 3),
            "goals": torch.tensor([[[0.5, 0.0, 0.0]]]),
            "estimated_contact": torch.tensor([0.0]),
        }


@pytest.mark.parametrize("limits", [None, [0.0, 0.4]])
def test_sensor_only_hold_and_reset(limits):
    def processor(images, **kwargs):
        return {"pixel_values": images[0].float(), "image_grid_thw": torch.ones(1, 3)}

    gate = {"mean": [0] * 20, "scale": [1] * 20, "coef": [0] * 20, "bias": 1}
    if limits is not None:
        gate["weights"] = limits
    weight = 1.0 if limits is None else limits[1]
    head = FixedHead()
    policy = ObservedController(head, QuadrantEncoder(), processor, gate=gate)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    state = np.zeros(8)
    for tick in range(5):
        state[0] = tick * 0.05
        cmd = policy.act(image, image, state)
        np.testing.assert_allclose(cmd[0], weight * (0.5 - state[0]) / (5 - tick), rtol=1e-6)
        np.testing.assert_array_equal(cmd[1:], [0, 0, 0, 0, 0, -1])
    assert head.calls == 1
    policy.act(image, image, state)
    assert head.calls == 2
    policy.reset()
    policy.act(image, image, state)
    assert head.calls == 3
