"""Independent binary gripper policy paired with a frozen six-axis arm policy."""

import torch
from torch import nn
from torch.nn import functional

from .modeling_demonstration_chunk import DemonstrationChunkHead


class GripperControlHead(nn.Module):
    def __init__(self, expert_config, mode):
        super().__init__()
        if mode not in {"grip_binary", "grip_event"}:
            raise ValueError(mode)
        self.mode = mode
        self.expert = DemonstrationChunkHead(**expert_config)
        width = expert_config["width"]
        self.camera, self.trunk = self.expert.encoder(expert_config["n_tasks"], width, 32)
        self.output = nn.Linear(width, 40 if mode == "grip_binary" else 41)

    def command(self, vision, state):
        proprio = (state[:, :8] - self.expert.state_mean) / self.expert.state_std
        encoded = torch.cat([self.camera(vision.float()).flatten(1), proprio, state[:, 8:]], dim=1)
        logits = self.output(self.trunk(encoded))
        if self.mode == "grip_binary":
            return logits
        cdf = functional.pad(logits[:, 1:].softmax(1)[:, :-1].cumsum(1), (1, 0))
        sign = (2 * logits[:, :1].sigmoid() - 1) * (1 - 2 * cdf)
        return torch.logit(((sign + 1) / 2).clamp(1e-6, 1 - 1e-6))

    def forward(self, vision, state):
        arm, _ = self.expert(vision, state)
        return torch.cat([arm[..., :6], self.command(vision, state)[..., None]], -1)

    def get_action(self, features, batch):
        return {"action_pred": self(features["backbone_features"], batch["state"][:, -1])}
