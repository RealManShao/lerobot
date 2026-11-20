"""Single-forward manipulation head over two camera views and robot proprioception."""

import torch
from torch import nn
from torch.nn import functional


def motion_coordinates(actions):
    """Squared distance weights arm command averages and gripper occupancy separately."""
    arm = actions[..., :6].unflatten(1, (-1, 5)).mean(2).flatten(1)
    gripper = actions[..., 6].unflatten(1, (-1, 5)).mean(2)
    return torch.cat([arm / arm.shape[1] ** 0.5, 0.35 * gripper / gripper.shape[1] ** 0.5], dim=1)


@torch.no_grad()
def fit_dictionary(actions, count):
    """Cluster within one task and retain real demonstration windows as representatives."""
    coords = motion_coordinates(actions)
    indices = [(coords - coords.mean(0)).square().sum(1).argmin()]
    nearest = torch.full_like(coords[:, 0], torch.inf)
    for _ in range(count - 1):
        nearest = torch.minimum(nearest, (coords - coords[indices[-1]]).square().sum(1))
        indices.append(nearest.argmax())
    indices = torch.stack(indices)
    for _ in range(8):
        labels = torch.cdist(coords, coords[indices]).argmin(1)
        for k in range(count):
            members = (labels == k).nonzero().flatten()
            if len(members):
                points = coords[members]
                indices[k] = members[(points - points.mean(0)).square().sum(1).argmin()]
    return actions[indices], torch.cdist(coords, coords[indices]).argmin(1)


EVENT_VARIANTS = {"event_time", "event_occupancy"}
DUAL_VARIANTS = {"dual_full", "dual_prefix", "separate_prefix"}
TEMPLATE_VARIANTS = {"medoid", "arm", "arm_grip", "soft_arm"} | DUAL_VARIANTS
VARIANTS = {"direct"} | EVENT_VARIANTS | TEMPLATE_VARIANTS


class DemonstrationChunkHead(nn.Module):
    def __init__(self, variant, n_tasks, count=24, width=384, horizon=40):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.horizon = horizon
        self.register_buffer("chunks", torch.zeros(n_tasks, count, horizon, 7))
        self.register_buffer("state_mean", torch.zeros(8))
        self.register_buffer("state_std", torch.ones(8))
        separate = variant == "separate_prefix"
        projection = 32 if separate else 64
        self.camera, self.trunk = self.encoder(n_tasks, width, projection)
        if separate:
            self.grip_camera, self.grip_trunk = self.encoder(n_tasks, width, projection)
        if variant in TEMPLATE_VARIANTS:
            self.selector = nn.Linear(width, count)
        if variant in DUAL_VARIANTS:
            self.arm_selector = nn.Linear(width, count)
        if variant in EVENT_VARIANTS:
            self.event = nn.Linear(width, horizon + 1)
        if variant != "medoid":
            dims = 7 if variant in {"direct", "arm_grip"} else 6
            self.correction = nn.Linear(width, dims * horizon)

    @staticmethod
    def encoder(n_tasks, width, projection):
        camera = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, projection), nn.GELU())
        trunk = nn.Sequential(
            nn.Linear(8 * projection + 8 + n_tasks, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        return camera, trunk

    def forward(self, vision, state):
        arm_state = (state[:, :8] - self.state_mean) / self.state_std
        task = state[:, 8:]
        h = self.trunk(torch.cat([self.camera(vision.float()).flatten(1), arm_state, task], dim=1))
        if self.variant == "direct":
            return self.correction(h).reshape(-1, self.horizon, 7), None
        if self.variant in EVENT_VARIANTS:
            event = self.event(h)
            # Bins 1..H-1 are switching steps; the last bin means no switch in the chunk.
            time_cdf = functional.pad(event[:, 1:].softmax(1)[:, :-1].cumsum(1), (1, 0))
            initial_sign = 2 * event[:, :1].sigmoid() - 1
            grip = initial_sign * (1 - 2 * time_cdf)
            return torch.cat([self.correction(h).reshape(-1, self.horizon, 6), grip[..., None]], -1), event
        grip_features = h
        if self.variant == "separate_prefix":
            grip_features = self.grip_trunk(
                torch.cat([self.grip_camera(vision.float()).flatten(1), arm_state, task], dim=1)
            )
        logits = self.selector(grip_features)
        available = self.chunks[task.argmax(1)]
        selected = available[torch.arange(len(h), device=h.device), logits.argmax(1)]
        if self.variant == "medoid":
            return selected, logits
        delta = self.correction(h).reshape(len(h), self.horizon, -1)
        arm_logits = self.arm_selector(h) if self.variant in DUAL_VARIANTS else logits
        if self.variant == "soft_arm" or self.variant in DUAL_VARIANTS:
            arm = torch.einsum("bk,bkhd->bhd", arm_logits.softmax(1), available[..., :6])
        else:
            arm = selected[..., :6]
        grip = selected[..., 6:]
        if self.variant == "arm_grip":
            grip = grip + delta[..., 6:]
        scores = (logits, arm_logits) if self.variant in DUAL_VARIANTS else logits
        return torch.cat([arm + delta[..., :6], grip], dim=-1), scores

    def get_action(self, features, batch):
        actions, _ = self(features["backbone_features"], batch["state"][:, -1])
        return {"action_pred": actions}

    def objective(self, vision, state, target, labels):
        actions, logits = self(vision, state)
        if self.variant == "event_time":
            grip = target[..., 6] > 0
            switched = grip[:, 1:] != grip[:, :-1]
            first = torch.where(switched.any(1), switched.long().argmax(1), self.horizon - 1)
            loss = functional.cross_entropy(
                logits[:, 1:], first
            ) + functional.binary_cross_entropy_with_logits(logits[:, 0], grip[:, 0].float())
        elif self.variant in DUAL_VARIANTS:
            grip_scores, arm_scores = logits
            loss = 0.5 * (
                functional.cross_entropy(grip_scores, labels[:, 0])
                + functional.cross_entropy(arm_scores, labels[:, 1])
            )
        elif self.variant == "event_occupancy":
            probability = ((actions[..., 6] + 1) * 0.5).clamp(1e-6, 1 - 1e-6)
            loss = functional.binary_cross_entropy(probability, (target[..., 6] > 0).float())
        elif self.variant == "direct":
            loss = actions.new_zeros(())
        else:
            loss = functional.cross_entropy(logits, labels)
        if self.variant != "medoid":
            loss = loss + 5 * functional.mse_loss(actions[..., :6], target[..., :6])
        if self.variant in {"arm_grip", "direct"}:
            loss = loss + functional.binary_cross_entropy_with_logits(
                actions[..., 6], (target[..., 6] > 0).float()
            )
        return loss
