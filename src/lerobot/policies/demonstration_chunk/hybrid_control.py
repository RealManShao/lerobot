"""One neural plan with a learned translation policy and a Cartesian waypoint servo."""

import torch
from torch import nn

from .modeling_demonstration_chunk import DemonstrationChunkHead


class HybridControlHead(nn.Module):
    modes = ("learned", "servo", "constant", "adaptive", "task", "hard")

    def __init__(self, expert_config, mode):
        super().__init__()
        if mode not in self.modes:
            raise ValueError(mode)
        self.mode = mode
        self.learner = DemonstrationChunkHead(**expert_config)
        tasks = expert_config["n_tasks"]
        self.camera, self.trunk = self.learner.encoder(tasks, 256, 32)
        self.waypoints = nn.Linear(256, 8 * 3)
        self.router = nn.Sequential(nn.Linear(256 + 18 + tasks, 64), nn.GELU(), nn.Linear(64, 1))
        self.register_buffer("motion", torch.eye(3))
        self.register_buffer("inverse_motion", torch.eye(3))
        self.register_buffer("step_scale", torch.ones(3))
        self.register_buffer("constant_weight", torch.tensor(0.5))
        self.register_buffer("task_weights", torch.full((tasks,), 0.5))

    def goals(self, vision, state):
        norm = (state[:, :8] - self.learner.state_mean) / self.learner.state_std
        features = self.trunk(torch.cat([self.camera(vision.float()).flatten(1), norm, state[:, 8:]], 1))
        displacement = self.waypoints(features).reshape(-1, 8, 3) * self.step_scale
        return features, displacement

    def experts(self, vision, state):
        action, _ = self.learner(vision, state)
        features, displacement = self.goals(vision, state)
        goal = state[:, None, :3] + displacement.cumsum(1)
        servo = (displacement @ self.inverse_motion / 5).clamp(-1, 1)
        arm = action[..., :3].unflatten(1, (8, 5))
        grip = action[..., 6].unflatten(1, (8, 5))
        context = torch.cat(
            [
                features[:, None].expand(-1, 8, -1),
                state[:, None, 8:].expand(-1, 8, -1),
                displacement / self.step_scale,
                servo,
                arm.mean(2),
                arm.std(2),
                grip[:, :, :1],
                grip[:, :, -1:],
                grip.mean(2, keepdim=True),
                state[:, None, 6:8].expand(-1, 8, -1) * 20,
                torch.linspace(0, 1, 8, device=state.device)[None, :, None].expand(len(state), -1, -1),
            ],
            -1,
        )
        return action, goal, servo, context

    def weights(self, context, state):
        if self.mode == "adaptive" or self.mode == "hard":
            w = self.router(context).sigmoid().squeeze(-1)
            return (w >= 0.5).float() if self.mode == "hard" else w
        if self.mode == "task":
            return self.task_weights[state[:, 8:].argmax(1), None].expand(-1, 8)
        value = {"learned": 0.0, "servo": 1.0, "constant": self.constant_weight}[self.mode]
        return torch.ones_like(context[..., 0]) * value

    def forward(self, vision, state):
        actions, goals, servo, context = self.experts(vision, state)
        w = self.weights(context, state)
        xyz = torch.lerp(
            actions[..., :3].clamp(-1, 1), servo.repeat_interleave(5, 1), w.repeat_interleave(5, 1)[..., None]
        )
        return {
            "action_pred": torch.cat([xyz, actions[..., 3:]], -1),
            "goals": goals,
            "weight": w,
            "learned_xyz": actions[..., :3],
        }

    def get_action(self, features, batch):
        return self(features["backbone_features"], batch["state"][:, -1])


class OutcomeControlHead(HybridControlHead):
    """Select a mixture using the measured effects of short simulator interventions."""

    outcome_modes = ("outcome_constant", "outcome_task", "outcome_phase", "outcome_context")

    def __init__(self, expert_config, mode, phase_classes):
        super().__init__(expert_config, "adaptive")
        if mode not in self.outcome_modes:
            raise ValueError(mode)
        self.mode = mode
        tasks = expert_config["n_tasks"]
        inputs = 256 + 18 + tasks
        self.phase_model = nn.Sequential(nn.Linear(inputs, 64), nn.GELU(), nn.Linear(64, phase_classes))
        self.cost_model = nn.Sequential(nn.Linear(inputs, 64), nn.GELU(), nn.Linear(64, 3))
        self.register_buffer("global_choice", torch.zeros((), dtype=torch.long))
        self.register_buffer("task_choice", torch.zeros(tasks, dtype=torch.long))
        self.register_buffer("phase_choice", torch.zeros(tasks, phase_classes, dtype=torch.long))

    def weights(self, context, state):
        task = state[:, 8:].argmax(1)
        if self.mode == "outcome_constant":
            choice = self.global_choice.expand(context.shape[:2])
        elif self.mode == "outcome_task":
            choice = self.task_choice[task, None].expand(context.shape[:2])
        elif self.mode == "outcome_phase":
            phase = self.phase_model(context).argmax(-1)
            choice = self.phase_choice[task[:, None], phase]
        else:
            choice = self.cost_model(context).argmin(-1)
        return 0.5 * choice.float()


class HistoryControlHead(HybridControlHead):
    """Update the first waypoint from two measured displacements at each neural replan."""

    history_modes = ("history_servo", "history_half", "history_constant", "history_adaptive")
    alignment_modes = ("history_aligned", "history_matched")

    def __init__(self, expert_config, mode):
        super().__init__(expert_config, "adaptive")
        if mode not in self.history_modes + self.alignment_modes:
            raise ValueError(mode)
        self.mode = mode
        self.register_buffer("history_fit", torch.zeros(9, 3))
        if mode in self.alignment_modes:
            self.register_buffer("goal_noise", torch.tensor(0.01))
        self.history_gate = nn.Sequential(
            nn.Linear(18 + expert_config["n_tasks"], 32), nn.GELU(), nn.Linear(32, 1)
        )

    def adjusted(self, vision, state, history):
        action, goals, _, _ = self.experts(vision, state)
        old = goals[:, 0] - state[:, :3]
        distance = torch.cat([old, history.flatten(1)], 1) @ self.history_fit
        updated = goals + (distance - old)[:, None]
        learned = action[:, :5, :3].clamp(-1, 1)
        gate_input = torch.cat(
            [
                old / self.step_scale,
                distance / self.step_scale,
                history.flatten(1) * 100,
                learned.mean(1),
                learned.std(1),
                state[:, 8:],
            ],
            1,
        )
        return action, updated, gate_input

    def alpha(self, features):
        if self.mode in {"history_adaptive", "history_aligned"}:
            weight = self.history_gate(features).sigmoid().flatten()
            if self.mode == "history_aligned":
                goal = features[:, :3] * self.step_scale
                ongoing = features[:, 6:9] * (5 / 100)
                progress = (ongoing * goal).sum(1) / (goal.square().sum(1) + self.goal_noise.square())
                weight = weight * progress.clamp(0, 1)
            return weight
        value = {
            "history_servo": 1.0,
            "history_half": 0.5,
            "history_constant": self.constant_weight,
            "history_matched": self.constant_weight,
        }[self.mode]
        return torch.ones_like(features[:, 0]) * value

    def forward(self, vision, state, history):
        action, goals, inputs = self.adjusted(vision, state, history)
        weight = self.alpha(inputs)[:, None].expand(-1, 8)
        origins = torch.cat([state[:, None, :3], goals[:, :-1]], 1)
        servo = ((goals - origins) @ self.inverse_motion / 5).clamp(-1, 1)
        xyz = torch.lerp(
            action[..., :3].clamp(-1, 1),
            servo.repeat_interleave(5, 1),
            weight.repeat_interleave(5, 1)[..., None],
        )
        return {
            "action_pred": torch.cat([xyz, action[..., 3:]], -1),
            "goals": goals,
            "weight": weight,
            "learned_xyz": action[..., :3],
        }

    def get_action(self, features, batch):
        return self(features["backbone_features"], batch["state"][:, -1], batch["displacements"])
