"""Execute a frozen visual action head using only synchronized robot observations."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .waypoint import waypoint_translation


def encode_rgb_batch(batch, visual, processor):
    """Pool a batch of 128-pixel training-format RGB views into four quadrant features."""
    pixels = processor(images=list(batch), return_tensors="pt", do_resize=False)
    device = next(visual.parameters()).device
    encoded = visual(
        pixels["pixel_values"].to(device=device, dtype=visual.dtype),
        grid_thw=pixels["image_grid_thw"].to(device),
    )
    return encoded.pooler_output.reshape(len(batch), 2, 2, 2, 2, -1).float().mean((2, 4)).flatten(1, 2).half()


def router_features(state, learned, initial_servo, goal, contact, grip):
    """Current proprioception and sensor-derived controller disagreement."""
    difference = initial_servo - learned.mean(axis=0)
    return np.concatenate(
        (
            state,
            goal - state[:3],
            learned.mean(axis=0),
            difference,
            [np.linalg.norm(difference), contact, grip],
        )
    )


class ObservedController:
    """A resettable 5-tick controller. Translation commands use the training controller's units."""

    def __init__(self, head, visual, processor, weight=0.0, gate=None):
        self.head = head.eval()
        self.visual = visual.eval()
        self.processor = processor
        self.weight = weight
        self.gate = gate
        self.inverse = head.inverse_motion.detach().cpu().numpy()
        self.reset()

    @classmethod
    def from_checkpoints(cls, vision_checkpoint, head_directory, switch_file=None, device="cuda"):
        """Load the RGB encoder and action head from disk. An optional switch file contains fitted coefficients."""
        from transformers import Qwen2VLImageProcessor

        from lerobot.policies.groot.modeling_groot import GrootPolicy

        from .visual_geometry import VisualGeometryHead

        root = Path(head_directory)
        run = json.loads((root / "object_constant.json").read_text())
        weights = root / run["weights"]
        head = VisualGeometryHead(**run["model"]).to(device)
        head.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        gate = None
        if switch_file is not None:
            spec = json.loads(Path(switch_file).read_text())
            if spec["head_sha256"] != hashlib.sha256(weights.read_bytes()).hexdigest():
                raise ValueError("Switch was fitted to different action-head weights")
            gate = spec["gate"]
        groot = GrootPolicy.from_pretrained(str(vision_checkpoint), local_files_only=True)
        visual = groot._groot_model.backbone.visual.to(device=device, dtype=torch.bfloat16)
        processor = Qwen2VLImageProcessor(
            patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3
        )
        return cls(head, visual, processor, head.global_w.item(), gate)

    def reset(self):
        self.tick = 0
        self.features = None
        self.last_weight = None

    @torch.inference_mode()
    def _plan(self, external_rgb, wrist_rgb, state):
        device = self.head.motion.device
        views = []
        for rgb in (external_rgb, wrist_rgb):
            if rgb.shape != (128, 128, 3) or rgb.dtype != np.uint8:
                raise ValueError("Expected a 128 x 128 uint8 RGB image per view")
            views.append(
                encode_rgb_batch(
                    torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None], self.visual, self.processor
                )
            )
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)[None]
        output = self.head(torch.cat(views, 1).float(), state_tensor)
        self.actions = output["action_pred"][0, :5].clamp(-1, 1).cpu().numpy()
        self.actions[:, 6] = np.where(self.actions[:, 6] > 0, 1.0, -1.0)
        self.learned = output["learned_xyz"][0, :5].clamp(-1, 1).cpu().numpy()
        self.goal = output["goals"][0, 0].cpu().numpy()
        initial_servo = waypoint_translation(state[:3], self.goal, self.inverse, 5)
        self.features = router_features(
            state,
            self.learned,
            initial_servo,
            self.goal,
            output["estimated_contact"].item(),
            self.actions[0, 6],
        )
        self.last_weight = self.weight
        if self.gate is not None:
            score = ((self.features - self.gate["mean"]) / self.gate["scale"]) @ self.gate[
                "coef"
            ] + self.gate["bias"]
            low, high = self.gate.get("weights", (0.0, 1.0))
            self.last_weight = high if score > 0 else low

    def act(self, external_rgb, wrist_rgb, state):
        """Return a 7-D command, replanning once every five calls."""
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Expected finite xyz, axis-angle and two finger positions")
        if self.tick == 0:
            self._plan(external_rgb, wrist_rgb, state)
        action = self.actions[self.tick].copy()
        servo = waypoint_translation(state[:3], self.goal, self.inverse, 5 - self.tick)
        nominal = self.learned[self.tick]
        action[:3] = nominal + self.last_weight * (servo - nominal)
        self.tick = (self.tick + 1) % 5
        return action
