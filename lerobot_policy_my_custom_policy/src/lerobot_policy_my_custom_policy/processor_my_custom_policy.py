# processor_my_custom_policy.py
from typing import Any

import torch
import torch.nn.functional as F

from lerobot.configs.types import NormalizationMode
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black.

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


@ProcessorStepRegistry.register(name="image_resize_processor_step")
class ImageResizeProcessorStep(ProcessorStep):
    """Processor step to resize images to the vision encoder's expected resolution."""

    def __init__(self, target_height: int = 224, target_width: int = 224, mode: str = "bilinear"):
        self.target_height = target_height
        self.target_width = target_width
        self.mode = mode

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Resize all image observations to target resolution."""
        transition = transition.copy()

        # Get observation keys
        observations = transition.get(TransitionKey.OBSERVATION, {})
        if not observations:
            return transition

        # Resize all image observations
        for key, value in observations.items():
            # Check if this is an image (has 3 or 4 dimensions and last dim is 3 or 4)
            if isinstance(value, torch.Tensor):
                if value.dim() == 3 and value.shape[-1] in [3, 4]:
                    # Single image (H, W, C)
                    observations[key] = resize_with_pad_torch(
                        value, self.target_height, self.target_width, self.mode
                    )
                elif value.dim() == 4 and value.shape[-1] in [3, 4]:
                    # Multiple images (T, H, W, C) or (B, H, W, C)
                    observations[key] = resize_with_pad_torch(
                        value, self.target_height, self.target_width, self.mode
                    )
                elif value.dim() == 5 and value.shape[-1] in [3, 4]:
                    # Batched temporal images (B, T, H, W, C)
                    B, T = value.shape[:2]
                    value_flat = value.view(B * T, *value.shape[2:])
                    resized_flat = resize_with_pad_torch(
                        value_flat, self.target_height, self.target_width, self.mode
                    )
                    observations[key] = resized_flat.view(B, T, *resized_flat.shape[1:])
                elif value.dim() == 4 and value.shape[1] in [3, 4]:
                    # Channels-first format (B, C, H, W) or (T, C, H, W)
                    observations[key] = resize_with_pad_torch(
                        value, self.target_height, self.target_width, self.mode
                    )
                elif value.dim() == 5 and value.shape[2] in [3, 4]:
                    # Batched temporal channels-first (B, T, C, H, W)
                    B, T = value.shape[:2]
                    value_flat = value.view(B * T, *value.shape[2:])
                    resized_flat = resize_with_pad_torch(
                        value_flat, self.target_height, self.target_width, self.mode
                    )
                    observations[key] = resized_flat.view(B, T, *resized_flat.shape[1:])

        transition[TransitionKey.OBSERVATION] = observations
        return transition

    def transform_features(self, features):
        """Update feature shapes to reflect the target resolution."""
        for feature_key, feature in features.items():
            if feature.type.value == "visual":
                # Update shape to target resolution
                if len(feature.shape) == 3:  # (C, H, W)
                    features[feature_key] = feature._replace(
                        shape=(feature.shape[0], self.target_height, self.target_width)
                    )
                elif len(feature.shape) == 4:  # (T, C, H, W)
                    features[feature_key] = feature._replace(
                        shape=(feature.shape[0], feature.shape[1], self.target_height, self.target_width)
                    )
        return features


@ProcessorStepRegistry.register(name="failure_point_processor_step")
class FailurePointProcessorStep(ProcessorStep):
    """
    Processor step to handle failure point labels for negative sample learning.

    This step ensures failure point labels are properly formatted and added to
    the complementary data for training.
    """

    def __init__(self, failure_label_key: str = "failure_point", default_value: int = -1):
        self.failure_label_key = failure_label_key
        self.default_value = default_value

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Ensure failure point labels are present in the transition."""
        transition = transition.copy()

        # Get complementary data
        if TransitionKey.COMPLEMENTARY_DATA not in transition:
            transition[TransitionKey.COMPLEMENTARY_DATA] = {}

        # Ensure failure point label exists
        if self.failure_label_key not in transition[TransitionKey.COMPLEMENTARY_DATA]:
            # Add default value (no failure)
            transition[TransitionKey.COMPLEMENTARY_DATA][self.failure_label_key] = self.default_value

        return transition

    def transform_features(self, features):
        """This step does not alter feature definitions."""
        return features


def make_my_custom_policy_pre_post_processors(
    config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for MyCustomPolicy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations
    2. Normalizing input and output features based on dataset statistics
    3. Adding a batch dimension
    4. Handling failure point labels for negative sample learning (if enabled)
    5. Moving all data to the specified device

    Note: Key frame selection is now handled by the model based on EEF speed,
    not by the processor. The processor provides all frames, and the model
    selects key frames dynamically during training/inference.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU
    2. Unnormalizing the output features to their original scale

    Args:
        config: The configuration object for MyCustomPolicy
        dataset_stats: A dictionary of statistics for normalization

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines
    """
    # Define normalization mapping
    normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
    }

    # Build input processing steps
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        ImageResizeProcessorStep(
            target_height=config.image_resolution[0],
            target_width=config.image_resolution[1],
        ),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    # Add failure point processor if negative sample learning is enabled
    if config.use_negative_samples:
        input_steps.append(
            FailurePointProcessorStep(
                failure_label_key=config.failure_label_key,
                default_value=-1,  # -1 indicates no failure
            )
        )

    # Move to device
    input_steps.append(DeviceProcessorStep(device=config.device))

    # Build output processing steps
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )