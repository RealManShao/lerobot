# modeling_my_custom_policy.py
import math
from collections import deque
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import _transformers_available
from .configuration_my_custom_policy import MyCustomPolicyConfig

# Conditional import for transformers
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from lerobot.policies.pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
    )
else:
    CONFIG_MAPPING = None
    PaliGemmaForConditionalGenerationWithPiGemma = None
    PiGemmaForCausalLM = None


def compute_eef_speed(poses: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """
    Compute end-effector speed from pose trajectory.

    Args:
        poses: (B, T, 7) - position (x,y,z) and quaternion (x,y,z,w)
        dt: Time step between consecutive poses

    Returns:
        speeds: (B, T-1) - speed at each timestep
    """
    B, T, _ = poses.shape

    # Extract positions (x, y, z)
    positions = poses[:, :, :3]  # (B, T, 3)

    # Compute displacements
    displacements = positions[:, 1:] - positions[:, :-1]  # (B, T-1, 3)

    # Compute speed (magnitude of displacement / dt)
    speeds = torch.norm(displacements, dim=-1) / dt  # (B, T-1)

    # Pad to match original length (use 0 for first timestep)
    speeds = F.pad(speeds, (0, 1), mode='constant', value=0.0)

    return speeds


def select_key_frames_by_speed(
    poses: torch.Tensor,
    dt: float,
    min_speed_threshold: float,
    max_speed_threshold: float,
    time_window: float,
    min_key_frames: int,
    max_key_frames: int,
) -> list[int]:
    """
    Select key frames based on EEF speed.

    High speed -> fewer key frames (movement is fast, accuracy less critical)
    Low speed -> more key frames (movement is slow, need more precision)

    Args:
        poses: (B, T, 7) - pose trajectory
        dt: Time step
        min_speed_threshold: Minimum speed to consider movement
        max_speed_threshold: Maximum speed for dense key frames
        time_window: Time window for key frame selection
        min_key_frames: Minimum number of key frames
        max_key_frames: Maximum number of key frames

    Returns:
        key_frame_indices: List of key frame indices
    """
    B, T, _ = poses.shape
    # Use first batch element for key frame selection
    speeds = compute_eef_speed(poses[0:1], dt)[0]  # (T,)

    # Average speed over the trajectory
    avg_speed = speeds.mean().item()

    # Determine number of key frames based on average speed
    # Higher speed -> fewer key frames
    # Lower speed -> more key frames
    if avg_speed < min_speed_threshold:
        # Very slow movement, use max key frames
        num_key_frames = max_key_frames
    elif avg_speed > max_speed_threshold:
        # Very fast movement, use min key frames
        num_key_frames = min_key_frames
    else:
        # Linear interpolation based on speed
        speed_ratio = (avg_speed - min_speed_threshold) / (max_speed_threshold - min_speed_threshold)
        num_key_frames = int(max_key_frames - speed_ratio * (max_key_frames - min_key_frames))
        num_key_frames = max(min_key_frames, min(max_key_frames, num_key_frames))

    # Convert time window to number of timesteps
    window_timesteps = int(time_window / dt)

    # Select key frames with minimum spacing
    key_frame_indices = [0]  # Always include start

    last_idx = 0
    while len(key_frame_indices) < num_key_frames and last_idx < T - 1:
        # Find next candidate index
        next_idx = min(last_idx + window_timesteps, T - 1)
        if next_idx == last_idx:
            next_idx = min(last_idx + 1, T - 1)

        key_frame_indices.append(next_idx)
        last_idx = next_idx

    # Ensure last frame is included
    if key_frame_indices[-1] != T - 1:
        key_frame_indices.append(T - 1)

    # Remove duplicates and sort
    key_frame_indices = sorted(list(set(key_frame_indices)))

    return key_frame_indices


class VisionEncoder(nn.Module):
    """SigLIP vision encoder from PaliGemma for image feature extraction."""

    def __init__(self, config: MyCustomPolicyConfig):
        super().__init__()
        self.config = config

        # Load PaliGemma configuration
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = config.gemma_hidden_size
        vlm_config_hf.text_config.intermediate_size = config.gemma_mlp_dim
        vlm_config_hf.text_config.num_attention_heads = config.gemma_num_heads
        vlm_config_hf.text_config.head_dim = config.gemma_head_dim
        vlm_config_hf.text_config.num_hidden_layers = config.gemma_num_layers
        vlm_config_hf.text_config.num_key_value_heads = config.gemma_num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.vision_config.image_size = config.image_resolution[0]
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        # Initialize PaliGemma model (only vision tower and projector)
        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)

        # Freeze vision encoder if requested
        if config.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Embed image using SigLIP vision encoder and multi-modal projector.

        Args:
            image: (B, C, H, W) - RGB image

        Returns:
            features: (B, num_patches, hidden_size) - Vision features projected to language model space
        """
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)

        # Extract image features using vision tower
        image_outputs = self.paligemma.model.get_image_features(image)

        # Apply multi-modal projector and scale
        features = image_outputs.pooler_output * self.paligemma.config.text_config.hidden_size**0.5

        if features.dtype != out_dtype:
            features = features.to(out_dtype)

        return features


class PoseEncoder(nn.Module):
    """Encode images and states at key frames using SigLIP + Gemma to predict 6D pose + gripper."""

    def __init__(self, config: MyCustomPolicyConfig):
        super().__init__()
        self.config = config

        # Vision encoder (SigLIP from PaliGemma)
        self.vision_encoder = VisionEncoder(config)

        # State encoder (for robot state if available)
        state_dim = config.input_features.get("observation.state", {}).get("shape", [32])[0]
        self.state_encoder = nn.Linear(state_dim, config.gemma_hidden_size)

        # Gemma language model for temporal processing
        gemma_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=config.gemma_head_dim,
            hidden_size=config.gemma_hidden_size,
            intermediate_size=config.gemma_mlp_dim,
            num_attention_heads=config.gemma_num_heads,
            num_hidden_layers=config.gemma_num_layers,
            num_key_value_heads=config.gemma_num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
        )

        # Use PiGemmaForCausalLM for the language model
        self.gemma_model = PiGemmaForCausalLM(config=gemma_config_hf)
        self.gemma_model.model.embed_tokens = None  # Remove token embeddings

        # Output head: 6D pose (position + quaternion) + gripper
        self.pose_head = nn.Sequential(
            nn.Linear(config.gemma_hidden_size, config.gemma_hidden_size),
            nn.ReLU(),
            nn.Linear(config.gemma_hidden_size, 7),  # 6D pose + gripper
        )

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        key_frame_indices: list[int],
    ) -> torch.Tensor:
        """
        Encode key frame observations and predict poses using SigLIP + Gemma.

        Args:
            images: (B, T, C, H, W) - images for all timesteps
            states: (B, T, state_dim) - states for all timesteps
            key_frame_indices: List of key frame indices

        Returns:
            key_frame_poses: (B, num_key_frames, 7) - predicted 6D pose + gripper at key frames
        """
        B, T, C, H, W = images.shape
        num_key_frames = len(key_frame_indices)

        # Extract key frame images and states
        key_images = images[:, key_frame_indices, :, :, :]  # (B, num_key_frames, C, H, W)
        key_states = states[:, key_frame_indices, :]  # (B, num_key_frames, state_dim)

        # Encode key frame images using SigLIP
        key_images_flat = key_images.view(B * num_key_frames, C, H, W)
        key_image_features = self.vision_encoder.embed_image(key_images_flat)  # (B*num_key_frames, num_patches, hidden_size)
        key_image_features = key_image_features.view(B, num_key_frames, -1, key_image_features.shape[-1])

        # Encode key frame states
        key_state_features = self.state_encoder(key_states)  # (B, num_key_frames, hidden_size)
        key_state_features = key_state_features.unsqueeze(1)  # (B, num_key_frames, 1, hidden_size)

        # Concatenate vision and state features
        # Flatten sequence: (B, num_key_frames, num_patches + 1, hidden_size)
        key_features = torch.cat([key_image_features, key_state_features], dim=2)

        # Flatten for Gemma: (B, num_key_frames * (num_patches + 1), hidden_size)
        key_features = key_features.view(B, -1, key_features.shape[-1])

        # Process through Gemma language model
        gemma_output = self.gemma_model(inputs_embeds=key_features)
        key_features = gemma_output.last_hidden_state  # (B, seq_len, hidden_size)

        # Extract features at state positions (last token per key frame)
        # Assuming state token is the last token for each key frame
        seq_len_per_keyframe = key_image_features.shape[2] + 1
        state_indices = torch.arange(
            seq_len_per_keyframe - 1, num_key_frames * seq_len_per_keyframe, seq_len_per_keyframe,
            device=key_features.device
        )
        key_features = key_features[:, state_indices, :]  # (B, num_key_frames, hidden_size)

        # Predict poses
        key_frame_poses = self.pose_head(key_features)  # (B, num_key_frames, 7)

        return key_frame_poses


def pose_distance_loss(pred_poses: torch.Tensor, target_poses: torch.Tensor) -> torch.Tensor:
    """
    Compute distance loss between predicted and target 6D poses + gripper.

    Args:
        pred_poses: (B, T, 7) - predicted (x,y,z, qx,qy,qz,qw, gripper)
        target_poses: (B, T, 7) - target (x,y,z, qx,qy,qz,qw, gripper)

    Returns:
        loss: (B, T) - loss for each timestep
    """
    # Position distance (L2)
    pred_pos = pred_poses[:, :, :3]
    target_pos = target_poses[:, :, :3]
    pos_loss = F.mse_loss(pred_pos, target_pos, reduction='none').sum(dim=-1)  # (B, T)

    # Orientation distance (quaternion angular distance)
    pred_quat = F.normalize(pred_poses[:, :, 3:7], dim=-1)
    target_quat = F.normalize(target_poses[:, :, 3:7], dim=-1)
    # Angular distance: 2 * arccos(|dot product|)
    dot_product = (pred_quat * target_quat).sum(dim=-1).clamp(-1.0, 1.0)
    angular_dist = 2 * torch.acos(torch.abs(dot_product))
    quat_loss = angular_dist ** 2  # (B, T)

    # Gripper loss (binary cross entropy or MSE)
    pred_gripper = pred_poses[:, :, 7:8]
    target_gripper = target_poses[:, :, 7:8]
    gripper_loss = F.mse_loss(pred_gripper, target_gripper, reduction='none').squeeze(-1)  # (B, T)

    # Combined loss
    total_loss = pos_loss + quat_loss + gripper_loss

    return total_loss


class MyCustomPolicyModel(nn.Module):
    """Core model for MyCustomPolicy with speed-based key frame selection."""

    def __init__(self, config: MyCustomPolicyConfig):
        super().__init__()
        self.config = config

        self.pose_encoder = PoseEncoder(config)

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        target_poses: torch.Tensor,
        key_frame_indices: list[int],
        failure_points: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
        dt: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with speed-based key frame selection.

        Args:
            images: (B, T, C, H, W)
            states: (B, T, state_dim)
            target_poses: (B, T, 7) - target 6D pose + gripper
            key_frame_indices: List of key frame indices
            failure_points: (B,) - timestep indices where failure occurred (-1 if no failure)
            action_is_pad: (B, T) - mask for padded actions
            dt: Time step duration

        Returns:
            dict with loss and predictions
        """
        B, T, _ = images.shape
        num_key_frames = len(key_frame_indices)

        # Predict poses at key frames
        pred_key_poses = self.pose_encoder(images, states, key_frame_indices)  # (B, num_key_frames, 7)

        # Extract target poses at key frames
        target_key_poses = target_poses[:, key_frame_indices, :]  # (B, num_key_frames, 7)

        # Compute pose distance loss at key frames
        key_frame_loss = pose_distance_loss(pred_key_poses, target_key_poses)  # (B, num_key_frames)

        # Apply negative sample learning if enabled
        if self.config.use_negative_samples and failure_points is not None:
            key_frame_loss = self.apply_negative_sample_learning(
                key_frame_loss, key_frame_indices, failure_points
            )

        # Reduce to scalar
        loss = key_frame_loss.mean()

        return {
            "loss": loss,
            "pred_key_poses": pred_key_poses,
            "target_key_poses": target_key_poses,
            "key_frame_indices": key_frame_indices,
        }

    def apply_negative_sample_learning(
        self,
        loss: torch.Tensor,
        key_frame_indices: list[int],
        failure_points: torch.Tensor,
    ) -> torch.Tensor:
        """Apply focal loss weighting to failure points."""
        B, num_key_frames = loss.shape

        # Create sample weights
        sample_weights = torch.ones_like(loss)  # (B, num_key_frames)

        # Weight failure points more heavily
        for b in range(B):
            failure_t = failure_points[b].item()
            if failure_t >= 0:
                # Find key frame closest to failure point
                closest_idx = min(key_frame_indices, key=lambda x: abs(x - failure_t))
                kf_idx = key_frame_indices.index(closest_idx)

                # Apply focal loss weighting
                error = loss[b, kf_idx]
                pt = torch.exp(-error)
                focal_weight = (1 - pt) ** self.config.focal_loss_gamma
                sample_weights[b, kf_idx] *= self.config.negative_sample_weight * focal_weight

        # Apply weights
        loss = loss * sample_weights

        return loss


class MyCustomPolicy(PreTrainedPolicy):
    config_class = MyCustomPolicyConfig
    name = "my_custom_policy"

    def __init__(self, config: MyCustomPolicyConfig, dataset_stats: dict[str, Any] = None):
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config

        # Initialize model
        self.model = MyCustomPolicyModel(config)

        self.reset()

    def reset(self):
        """Reset episode state."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._current_key_frame_idx = 0
        self._key_frame_poses = None
        self._key_frame_indices = None

    def get_optim_params(self) -> dict:
        """Return parameters to pass to the optimizer."""
        return {"params": self.parameters()}

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return the full action chunk (B, chunk_size, action_dim) for the current observation."""
        self.eval()

        # Extract images and states from batch
        images = self._extract_images(batch)
        states = self._extract_states(batch)

        # Get poses from batch (assuming action contains pose information)
        poses = self._extract_poses(batch)

        with torch.no_grad():
            B = images.shape[0]
            T = images.shape[1]
            dt = 0.1  # Default time step

            # Select key frames based on speed
            key_frame_indices = select_key_frames_by_speed(
                poses,
                dt,
                self.config.min_speed_threshold,
                self.config.max_speed_threshold,
                self.config.time_window,
                self.config.min_key_frames,
                self.config.max_key_frames,
            )

            # Predict poses at key frames
            pred_key_poses = self.model.pose_encoder(images, states, key_frame_indices)

            # Store for interpolation
            self._key_frame_poses = pred_key_poses[0].cpu()  # Store first batch element
            self._key_frame_indices = key_frame_indices
            self._current_key_frame_idx = 0

        # Convert poses to actions (implementation depends on action space)
        # For now, return poses as actions
        actions = self._poses_to_actions(pred_key_poses)

        # Return first n_action_steps
        return actions[:, : self.config.n_action_steps, :]

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return a single action for the current timestep (called at inference)."""
        # Action queue logic
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the training loss."""
        # Extract images and states
        images = self._extract_images(batch)
        states = self._extract_states(batch)

        # Get target poses (assuming action contains pose information)
        target_poses = self._extract_poses(batch)

        # Get failure points if available
        failure_points = batch.get(self.config.failure_label_key)

        # Get padding mask
        action_is_pad = batch.get("action_is_pad")

        # Select key frames based on speed
        dt = 0.1  # Default time step
        key_frame_indices = select_key_frames_by_speed(
            target_poses,
            dt,
            self.config.min_speed_threshold,
            self.config.max_speed_threshold,
            self.config.time_window,
            self.config.min_key_frames,
            self.config.max_key_frames,
        )

        # Forward pass
        output = self.model(
            images, states, target_poses, key_frame_indices, failure_points, action_is_pad, dt
        )

        return {"loss": output["loss"]}

    def _extract_images(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract and stack images from batch."""
        # Get image feature keys
        image_keys = [k for k in self.config.image_features if k in batch]

        if not image_keys:
            raise ValueError("No image features found in batch")

        # For simplicity, use the first image feature
        images = batch[image_keys[0]]  # (B, T, C, H, W) or (B, C, H, W)

        # Add time dimension if missing
        if images.dim() == 4:
            images = images.unsqueeze(1).expand(-1, self.config.horizon, -1, -1, -1)

        return images

    def _extract_states(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract and prepare states from batch."""
        state_key = "observation.state"
        if state_key not in batch:
            # Create dummy states if not available
            B = batch[ACTION].shape[0]
            state_dim = self.config.input_features.get(state_key, {}).get("shape", [32])[0]
            states = torch.zeros(B, self.config.horizon, state_dim, device=batch[ACTION].device)
        else:
            states = batch[state_key]
            # Add time dimension if missing
            if states.dim() == 2:
                states = states.unsqueeze(1).expand(-1, self.config.horizon, -1)

        return states

    def _extract_poses(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract 6D pose + gripper from actions.

        Assumes actions contain position (3), orientation (4 as quaternion), and gripper (1).
        If actions have different structure, this needs to be adapted.
        """
        actions = batch[ACTION]  # (B, T, action_dim)

        B, T, action_dim = actions.shape

        # If action_dim is 7, assume it's already pose + gripper
        if action_dim == 7:
            return actions
        # If action_dim is different, need to extract or transform
        # For now, assume first 7 dimensions are pose + gripper
        elif action_dim >= 7:
            return actions[:, :, :7]
        else:
            # Pad with zeros if action_dim < 7
            poses = F.pad(actions, (0, 7 - action_dim), mode='constant', value=0.0)
            return poses

    def _poses_to_actions(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Convert poses to actions.

        This is a placeholder - implementation depends on the action space.
        """
        # For now, return poses directly
        return poses