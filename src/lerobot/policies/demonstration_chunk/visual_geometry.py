"""A task-blind manipulation policy with vision-based entity pose and contact estimates."""

import torch
from torch import nn
from torch.nn import functional

from .modeling_demonstration_chunk import DemonstrationChunkHead


def rotation(frame):
    x = functional.normalize(frame[..., :3], dim=-1)
    y = frame[..., 3:]
    y = functional.normalize(y - (x * y).sum(-1, keepdim=True) * x, dim=-1)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], -1)


class VisualGeometryHead(nn.Module):
    routes = ("direct", "half", "constant", "contact")

    def __init__(self, geometry, route="direct"):
        super().__init__()
        if geometry not in {"world", "object", "consistent", "shared"} or route not in (
            *self.routes,
            "entity_contact",
        ):
            raise ValueError((geometry, route))
        self.geometry, self.route = geometry, route
        if route == "entity_contact":
            self.register_buffer("entity_w", torch.zeros(4, 2))
        width = 485 if geometry == "shared" else 384
        self.camera, self.trunk = DemonstrationChunkHead.encoder(0, width, 64)
        if geometry != "shared":
            self.grip_camera, self.grip_trunk = DemonstrationChunkHead.encoder(0, 224, 32)
        self.arm = nn.Linear(width, 240)
        self.grip = nn.Linear(width if geometry == "shared" else 224, 40)
        self.entities = nn.Linear(width, 4 * 11)
        self.anchors = nn.Linear(width, 4)
        self.offset = nn.Linear(width, 12 if geometry in {"object", "consistent"} else 3)
        for k, x in {
            "state_mean": torch.zeros(8),
            "state_std": torch.ones(8),
            "center": torch.zeros(3),
            "motion": torch.eye(3),
            "inverse_motion": torch.eye(3),
            "global_w": torch.zeros(()),
            "stage_w": torch.zeros(2),
        }.items():
            self.register_buffer(k, x)

    def predict(self, vision, state):
        if state.shape[-1] != 8:
            raise ValueError("Expected only end-effector and gripper proprioception")
        proprio = (state - self.state_mean) / self.state_std
        h = self.trunk(torch.cat([self.camera(vision.float()).flatten(1), proprio], -1))
        g = (
            h
            if self.geometry == "shared"
            else self.grip_trunk(torch.cat([self.grip_camera(vision.float()).flatten(1), proprio], -1))
        )
        slots = self.entities(h).reshape(-1, 4, 11)
        pos = self.center + 0.18 * slots[..., :3]
        rot = rotation(slots[..., 3:9])
        valid, touch = slots[..., 9], slots[..., 10]
        assoc = self.anchors(h) + functional.logsigmoid(valid)
        local = self.offset(h).reshape(len(h), -1, 3)
        if self.geometry in {"object", "consistent"}:
            local = 0.15 * local
            candidates = pos + (rot @ local.unsqueeze(-1)).squeeze(-1)
            goal = (assoc.softmax(1)[..., None] * candidates).sum(1)
        else:
            goal = state[:, :3] + 0.04 * local[:, 0]
        contact = (touch.sigmoid() * valid.sigmoid()).amax(1)
        return (
            self.arm(h).reshape(-1, 40, 6),
            self.grip(g),
            goal,
            pos,
            rot,
            valid,
            touch,
            assoc,
            contact,
            local,
        )

    @staticmethod
    def entity_stage(assoc, valid, logit):
        contact = logit.sigmoid() * valid.sigmoid()
        return assoc.softmax(1)[..., None] * torch.stack([1 - contact, contact], dim=-1)

    def forward(self, vision, state):
        arm, grip, goal, pos, rot, valid, logit, assoc, touch, _ = self.predict(vision, state)
        if self.route == "entity_contact":
            design = self.entity_stage(assoc, valid, logit)
            w = (design * self.entity_w).sum((1, 2))
        elif self.route == "contact":
            w = torch.lerp(self.stage_w[0], self.stage_w[1], touch)
        elif self.route == "constant":
            w = self.global_w.expand(len(state))
        else:
            w = torch.full_like(touch, 0.5 if self.route == "half" else 0.0)
        nominal = arm.clone()
        servo = ((goal - state[:, :3]) @ self.inverse_motion / 5).clamp(-1, 1)
        nominal[:, :5, :3] = torch.lerp(arm[:, :5, :3].clamp(-1, 1), servo[:, None], w[:, None, None])
        return {
            "action_pred": torch.cat([nominal, grip[..., None]], -1),
            "weight": w[:, None],
            "goals": goal[:, None],
            "learned_xyz": arm[..., :3],
            "estimated_contact": touch,
            "positions": pos,
            "rotations": rot,
        }

    def get_action(self, features, batch):
        return self(features["backbone_features"], batch["state"][:, -1])
