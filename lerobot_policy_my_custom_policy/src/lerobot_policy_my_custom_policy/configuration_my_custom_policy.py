# configuration_my_custom_policy.py
from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

DEFAULT_IMAGE_SIZE = 224

@PreTrainedConfig.register_subclass("my_custom_policy")
@dataclass
class MyCustomPolicyConfig(PreTrainedConfig):
    """Configuration class for MyCustomPolicy with key frame and negative sample learning.

    Args:
        n_obs_steps: Number of observation steps to use as input
        horizon: Action prediction horizon (total steps in episode)
        n_action_steps: Number of action steps to execute at inference
        hidden_dim: Hidden dimension for the policy network (deprecated, use gemma_hidden_size)
        min_speed_threshold: Minimum speed to consider movement
        max_speed_threshold: Maximum speed for dense key frames
        time_window: Time window in seconds for key frame selection
        min_key_frames: Minimum number of key frames (start and end)
        max_key_frames: Maximum number of key frames to prevent over-segmentation
        use_negative_samples: Enable learning from negative (near-failure) samples
        negative_sample_weight: Weight for negative sample loss
        failure_label_key: Key in batch for failure point labels
        focal_loss_gamma: For focal loss on failure points
        vision_encoder_name: Name of the vision encoder model (e.g., "google/paligemma-3b-pt-224")
        image_resolution: Image resolution for vision encoder (height, width)
        freeze_vision_encoder: Whether to freeze the vision encoder during training
        gemma_variant: Gemma variant to use for language model ("gemma_2b" or "gemma_300m")
        gemma_hidden_size: Hidden size for Gemma language model
        gemma_num_layers: Number of layers in Gemma language model
        gemma_num_heads: Number of attention heads in Gemma language model
        gemma_num_kv_heads: Number of key-value heads in Gemma language model
        gemma_head_dim: Head dimension for Gemma language model
        gemma_mlp_dim: MLP dimension for Gemma language model
        dtype: Data type for model ("bfloat16" or "float32")
    """

    horizon: int = 50
    n_action_steps: int = 50
    hidden_dim: int = 256  # Deprecated, use gemma_hidden_size

    # Vision encoder configuration
    vision_encoder_name: str = "google/paligemma-3b-pt-224"
    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    freeze_vision_encoder: bool = False

    # Gemma language model configuration
    gemma_variant: str = "gemma_2b"
    gemma_hidden_size: int = 2048
    gemma_num_layers: int = 18
    gemma_num_heads: int = 8
    gemma_num_kv_heads: int = 1
    gemma_head_dim: int = 256
    gemma_mlp_dim: int = 16384
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    # Speed-based key frame selection parameters
    min_speed_threshold: float = 0.01  # Minimum speed to consider movement
    max_speed_threshold: float = 2.0   # Maximum speed for dense key frames
    time_window: float = 0.5  # Time window in seconds for key frame selection
    min_key_frames: int = 2  # Minimum number of key frames (start and end)
    max_key_frames: int = 20  # Maximum number of key frames to prevent over-segmentation

    # Negative sample learning configuration
    use_negative_samples: bool = True
    negative_sample_weight: float = 2.0  # Higher weight for failure points
    failure_label_key: str = "failure_point"  # Key in batch for failure timestep indices
    focal_loss_gamma: float = 2.0  # For focal loss on failure points

    # Optimizer settings
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps cannot exceed horizon")
        if self.min_key_frames < 2:
            raise ValueError(f"min_key_frames must be at least 2, got {self.min_key_frames}")
        if self.max_key_frames < self.min_key_frames:
            raise ValueError(f"max_key_frames must be >= min_key_frames")
        if self.time_window <= 0:
            raise ValueError(f"time_window must be positive, got {self.time_window}")
        if self.gemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid gemma_variant: {self.gemma_variant}")
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.image_resolution[0] != self.image_resolution[1]:
            raise ValueError(
                f"SigLIP expects square image resolution, invalid resolution: {self.image_resolution}"
            )

    def validate_features(self) -> None:
        """Validate input/output feature compatibility."""
        if not self.image_features:
            raise ValueError("MyCustomPolicy requires at least one image feature.")
        if self.action_feature is None:
            raise ValueError("MyCustomPolicy requires 'action' in output_features.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> list[int] | None:
        """Relative timestep offsets the dataset loader provides per observation.

        Return `None` for single-frame policies. For temporal policies that consume
        multiple past or future frames, return a list of offsets, e.g. `[-20, -10, 0, 10]` for
        3 past frames at stride 10 and 1 future frame at stride 10.
        """
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Relative timestep offsets for the action chunk the dataset loader returns.
        
        When using key frames, this returns indices for all steps, but the model
        will focus on key frames during training.
        """
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None