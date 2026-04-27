# Model Architecture Documentation

This document provides a detailed technical description of the MyCustomPolicy model architecture, including layer specifications, sub-module details, and data flow.

## Overview

MyCustomPolicy is a Vision-Language-Action (VLA) policy that predicts 6D end-effector poses (position + orientation) and gripper values at key frames selected based on end-effector movement speed. The model consists of three main components:

1. **Speed-Based Key Frame Selection**: Automatically determines key frames from the trajectory based on EEF speed
2. **PoseEncoder**: Encodes visual and state observations at key frames to predict 6D poses
3. **Pose Distance Loss**: Computes loss based on position, orientation, and gripper distances

## Architecture Diagram

```
Input: (B, T, C, H, W) images, (B, T, state_dim) states, (B, T, 7) target poses
         |
         v
+---------------------------+
| Speed-Based Key Frame     |
| Selection                 |
| - compute_eef_speed()     |
| - select_key_frames_by... |
+---------------------------+
         |
         v
Key Frame Indices: [t1, t2, ..., tN]
         |
         v
+---------------------------+
| PoseEncoder               |
|                           |
| +---------------------+   |
| | Image Encoder       |   |
| | - Conv2d (3->64)    |   |
| | - ReLU              |   |
| | - MaxPool2d         |   |
| | - Conv2d (64->128)  |   |
| | - ReLU              |   |
| | - AdaptiveAvgPool2d |   |
| | - Flatten           |   |
| | - Linear (128->D)   |   |
| +---------------------+   |
|           |               |
| +---------------------+   |
| | State Encoder      |   |
| | - Linear (S->D)    |   |
| +---------------------+   |
|           |               |
|           v               |
| +---------------------+   |
| | Fusion Layer        |   |
| | - Linear (2D->D)   |   |
| +---------------------+   |
|           |               |
|           v               |
| +---------------------+   |
| | Temporal Encoder    |   |
| | - TransformerEnc    |   |
| |   (4 layers)       |   |
| +---------------------+   |
|           |               |
|           v               |
| +---------------------+   |
| | Pose Head           |   |
| | - Linear (D->D)     |   |
| | - ReLU              |   |
| | - Linear (D->7)     |   |
| +---------------------+   |
+---------------------------+
         |
         v
Output: (B, N, 7) predicted poses at key frames
         |
         v
+---------------------------+
| Pose Distance Loss        |
| - Position L2 loss        |
| - Quaternion angular loss |
| - Gripper MSE loss        |
+---------------------------+
         |
         v
Total Loss (with optional focal loss weighting for failure points)
```

## 1. Speed-Based Key Frame Selection

### 1.1 `compute_eef_speed()`

Computes the end-effector speed from a pose trajectory.

**Input:**
- `poses`: Tensor of shape `(B, T, 7)` containing position (x,y,z) and quaternion (x,y,z,w)
- `dt`: Time step between consecutive poses (default: 0.1s)

**Output:**
- `speeds`: Tensor of shape `(B, T)` containing speed at each timestep

**Algorithm:**
1. Extract positions: `positions = poses[:, :, :3]` → shape `(B, T, 3)`
2. Compute displacements: `displacements = positions[:, 1:] - positions[:, :-1]` → shape `(B, T-1, 3)`
3. Compute speed magnitude: `speeds = ||displacements|| / dt` → shape `(B, T-1)`
4. Pad with zeros at beginning to match original length: shape `(B, T)`

### 1.2 `select_key_frames_by_speed()`

Selects key frame indices based on average end-effector speed.

**Input:**
- `poses`: Tensor of shape `(B, T, 7)` - pose trajectory
- `dt`: Time step duration
- `min_speed_threshold`: Minimum speed to consider movement (default: 0.01)
- `max_speed_threshold`: Maximum speed for dense key frames (default: 2.0)
- `time_window`: Time window in seconds for key frame spacing (default: 0.5)
- `min_key_frames`: Minimum number of key frames (default: 2)
- `max_key_frames`: Maximum number of key frames (default: 20)

**Output:**
- `key_frame_indices`: List of integers representing key frame timestep indices

**Algorithm:**
1. Compute average speed over trajectory using `compute_eef_speed()`
2. Determine number of key frames based on speed:
   - If `avg_speed < min_speed_threshold`: Use `max_key_frames` (slow movement → more precision)
   - If `avg_speed > max_speed_threshold`: Use `min_key_frames` (fast movement → fewer frames)
   - Otherwise: Linearly interpolate between min and max based on speed ratio
3. Convert time window to timesteps: `window_timesteps = int(time_window / dt)`
4. Select key frames with minimum spacing:
   - Always include index 0 (start)
   - Add subsequent frames with at least `window_timesteps` gap
   - Always include last index (end)
5. Remove duplicates and sort

**Key Insight:** High-speed movements use fewer key frames because accuracy is less critical when moving fast. Low-speed movements use more key frames for precision during delicate operations.

## 2. PoseEncoder Architecture

The PoseEncoder processes observations at key frames to predict 6D poses. It consists of four main sub-modules.

### 2.1 Vision Encoder (SigLIP from PaliGemma)

Encodes visual observations using SigLIP vision encoder from PaliGemma, with multi-modal projector.

**Architecture:**
- **Vision Tower**: SigLIP vision encoder (from PaliGemma-3b-pt-224)
  - Input: `(B, 3, 224, 224)` - RGB images at 224x224 resolution
  - Output: Vision features (patch embeddings)
  - Pretrained on large-scale image-text datasets

- **Multi-Modal Projector**: Projects vision features to language model space
  - Input: Vision features from SigLIP
  - Output: `(B, num_patches, hidden_size)` where `hidden_size` is 2048 (for gemma_2b)
  - Uses GELU activation
  - Scales features by `sqrt(hidden_size)` for normalization

**Configuration:**
```python
vision_config.image_size = 224
vision_config.intermediate_size = 4304
vision_config.projection_dim = 2048
vision_config.projector_hidden_act = "gelu_fast"
vision_config.dtype = "float32"
```

**Key Features:**
- Vision tower operates in float32 for numerical stability
- Multi-modal projector aligns vision features with language model embedding space
- Supports freezing vision encoder during training (`freeze_vision_encoder` config option)
- Images are automatically resized to 224x224 with padding by the processor

**Parameters:** ~400M (pretrained SigLIP + projector) - typically frozen during fine-tuning

### 2.2 State Encoder

Encodes robot state (joint positions, velocities, etc.) into feature vectors aligned with language model space.

**Layers:**
1. `nn.Linear(state_dim, gemma_hidden_size)`
   - Input: `(B, num_key_frames, state_dim)` where `state_dim` is typically 32
   - Output: `(B, num_key_frames, gemma_hidden_size)` where `gemma_hidden_size` is 2048 (for gemma_2b)
   - Parameters: state_dim * gemma_hidden_size + gemma_hidden_size

**Purpose:** Projects state features to the same dimension as vision features for concatenation.

**Note:** If state is not available, dummy states are created with zeros.

### 2.3 Feature Concatenation

Combines vision and state features into a sequence for Gemma language model processing.

**Process:**
1. Vision features: `(B, num_key_frames, num_patches, gemma_hidden_size)`
2. State features: `(B, num_key_frames, 1, gemma_hidden_size)` (unsqueezed)
3. Concatenation: `torch.cat([vision_features, state_features], dim=2)`
   - Output: `(B, num_key_frames, num_patches + 1, gemma_hidden_size)`

**Flattening:** The sequence is flattened for Gemma processing:
- Input to Gemma: `(B, num_key_frames * (num_patches + 1), gemma_hidden_size)`

**Purpose:** Creates a token sequence where each key frame contributes multiple tokens (vision patches + state token) for temporal processing.

### 2.4 Gemma Language Model

Processes the concatenated vision-state sequence using Gemma language model for temporal understanding.

**Architecture:**
- **Model:** PiGemmaModel (Gemma with AdaRMS and gated residuals)
- **Variant:** gemma_2b (default) or gemma_300m
- **Configuration (gemma_2b):**
  - Hidden size: 2048
  - Number of layers: 18
  - Number of attention heads: 8
  - Number of key-value heads: 1 (grouped-query attention)
  - Head dimension: 256
  - MLP dimension: 16384
  - Activation: GELU (gelu_pytorch_tanh)

**Input:** `(B, seq_len, gemma_hidden_size)` where seq_len = num_key_frames * (num_patches + 1)
**Output:** `(B, seq_len, gemma_hidden_size)` - Processed sequence

**Key Features:**
- Uses rotary position embeddings for positional information
- Supports grouped-query attention for efficiency
- AdaRMS (adaptive RMSNorm) with gated residuals for conditional modulation
- Token embeddings are removed (inputs provided as embeddings directly)

**Parameters:**
- gemma_2b: ~2B parameters
- gemma_300m: ~300M parameters

**Purpose:** Captures complex temporal dependencies and relationships between key frames using a powerful language model backbone.

### 2.5 Feature Extraction

Extracts the state token features from Gemma output for each key frame.

**Process:**
- After Gemma processing, extract features at state token positions
- State tokens are at indices: `seq_len_per_keyframe - 1, 2*seq_len_per_keyframe - 1, ...`
- Output: `(B, num_key_frames, gemma_hidden_size)`

**Purpose:** Retrieves the processed representation for each key frame (state token aggregates vision information through attention).

### 2.6 Pose Head

Predicts 6D pose (position + quaternion) and gripper value.

**Layers:**
1. `nn.Linear(gemma_hidden_size, gemma_hidden_size)`
   - Input: `(B, num_key_frames, gemma_hidden_size)`
   - Output: `(B, num_key_frames, gemma_hidden_size)`

2. `nn.ReLU()`
   - Activation function

3. `nn.Linear(gemma_hidden_size, 7)`
   - Output: `(B, num_key_frames, 7)` - [x, y, z, qx, qy, qz, qw, gripper]
   - Parameters: gemma_hidden_size * 7 + 7

**Output Interpretation:**
- First 3 values: Position (x, y, z)
- Next 4 values: Quaternion (qx, qy, qz, qw) representing orientation
- Last value: Gripper state (binary or continuous)

## 3. Pose Distance Loss

Computes the distance loss between predicted and target 6D poses + gripper.

### 3.1 Position Loss

**Formula:** L2 distance between predicted and target positions

```python
pred_pos = pred_poses[:, :, :3]  # (B, T, 3)
target_pos = target_poses[:, :, :3]  # (B, T, 3)
pos_loss = F.mse_loss(pred_pos, target_pos, reduction='none').sum(dim=-1)  # (B, T)
```

**Loss:** Sum of squared differences in x, y, z coordinates

### 3.2 Orientation Loss

**Formula:** Quaternion angular distance

```python
pred_quat = F.normalize(pred_poses[:, :, 3:7], dim=-1)  # (B, T, 4)
target_quat = F.normalize(target_poses[:, :, 3:7], dim=-1)  # (B, T, 4)
dot_product = (pred_quat * target_quat).sum(dim=-1).clamp(-1.0, 1.0)
angular_dist = 2 * torch.acos(torch.abs(dot_product))
quat_loss = angular_dist ** 2  # (B, T)
```

**Loss:** Squared angular distance between quaternions

**Key Points:**
- Quaternions are normalized to unit length
- Absolute value of dot product handles quaternion double-cover (q and -q represent same rotation)
- Arcosine gives angle in radians
- Squared for loss computation

### 3.3 Gripper Loss

**Formula:** Mean squared error for gripper value

```python
pred_gripper = pred_poses[:, :, 7:8]  # (B, T, 1)
target_gripper = target_poses[:, :, 7:8]  # (B, T, 1)
gripper_loss = F.mse_loss(pred_gripper, target_gripper, reduction='none').squeeze(-1)  # (B, T)
```

**Loss:** Squared difference in gripper state

### 3.4 Total Loss

```python
total_loss = pos_loss + quat_loss + gripper_loss  # (B, T)
```

The loss is computed at key frames only, then averaged across batch and key frames.

## 4. Negative Sample Learning

### 4.1 Focal Loss Weighting

Applies higher loss weights around failure points using focal loss.

**Algorithm:**
1. For each batch element with a failure point (failure_t >= 0):
   - Find the key frame closest to the failure timestep
   - Compute focal weight: `(1 - pt)^gamma` where `pt = exp(-error)`
   - Apply weight: `sample_weights[b, kf_idx] *= negative_sample_weight * focal_weight`

2. Apply weights to loss: `loss = loss * sample_weights`

**Key Insight:** Focal loss focuses learning on hard examples (high error) by upweighting them. Failure points receive additional weighting via `negative_sample_weight`.

**Parameters:**
- `negative_sample_weight`: Base multiplier for failure point loss (default: 2.0)
- `focal_loss_gamma`: Focusing parameter (default: 2.0)
  - Higher gamma → more aggressive focusing on hard examples

## 5. Training Flow

### 5.1 Forward Pass (Training)

```
Input Batch
  ↓
Extract images, states, target poses, failure_points
  ↓
Select key frames by speed (using target poses)
  ↓
PoseEncoder.forward(images, states, key_frame_indices)
  ↓
Predict key frame poses
  ↓
Extract target poses at key frames
  ↓
Compute pose_distance_loss
  ↓
Apply negative sample learning (if enabled)
  ↓
Return mean loss
```

### 5.2 Inference Flow

```
Input Batch (single observation)
  ↓
Extract images, states, poses
  ↓
Select key frames by speed (using poses)
  ↓
PoseEncoder.forward(images, states, key_frame_indices)
  ↓
Predict key frame poses
  ↓
Store key frame poses and indices
  ↓
Convert poses to actions
  ↓
Return first n_action_steps actions
  ↓
Subsequent calls: Pop from action queue
```

## 6. Parameter Summary

### 6.1 PoseEncoder Parameters (for gemma_2b variant)

| Component | Parameters |
|-----------|------------|
| Vision Encoder (SigLIP) | ~400M (pretrained, typically frozen) |
| State Encoder | state_dim * 2048 + 2048 ≈ 65,792 (for state_dim=32) |
| Gemma Language Model | ~2B (gemma_2b) or ~300M (gemma_300m) |
| Pose Head | 2048 * 2048 + 2048 + 2048 * 7 + 7 = 4,203,527 |
| **Total (trainable)** | **~2.3B (gemma_2b) or ~300M (gemma_300m)** |

**Note:** If `freeze_vision_encoder=True`, the SigLIP parameters are frozen, reducing trainable parameters by ~400M.

### 6.2 Configurable Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Vision Encoder | | | |
| `vision_encoder_name` | str | "google/paligemma-3b-pt-224" | HuggingFace model name for vision encoder |
| `image_resolution` | tuple | (224, 224) | Image resolution (height, width) |
| `freeze_vision_encoder` | bool | False | Whether to freeze vision encoder |
| Gemma Language Model | | | |
| `gemma_variant` | str | "gemma_2b" | Gemma variant ("gemma_2b" or "gemma_300m") |
| `gemma_hidden_size` | int | 2048 | Hidden size for Gemma |
| `gemma_num_layers` | int | 18 | Number of layers in Gemma |
| `gemma_num_heads` | int | 8 | Number of attention heads |
| `gemma_num_kv_heads` | int | 1 | Number of key-value heads |
| `gemma_head_dim` | int | 256 | Head dimension |
| `gemma_mlp_dim` | int | 16384 | MLP dimension |
| `dtype` | str | "float32" | Data type ("float32" or "bfloat16") |
| Key Frame Selection | | | |
| `min_speed_threshold` | float | 0.01 | Minimum speed for key frame selection |
| `max_speed_threshold` | float | 2.0 | Maximum speed for key frame selection |
| `time_window` | float | 0.5 | Time window for key frame spacing |
| `min_key_frames` | int | 2 | Minimum number of key frames |
| `max_key_frames` | int | 20 | Maximum number of key frames |
| Training | | | |
| `use_negative_samples` | bool | True | Enable negative sample learning |
| `negative_sample_weight` | float | 2.0 | Weight for failure point loss |
| `focal_loss_gamma` | float | 2.0 | Focal loss gamma parameter |
| `horizon` | int | 50 | Action prediction horizon |
| `n_action_steps` | int | 50 | Action steps to execute at inference |
| `hidden_dim` | int | 256 | Deprecated, use gemma_hidden_size |

## 7. Key Design Decisions

### 7.1 Speed-Based Key Frame Selection

**Rationale:** Different movement speeds require different levels of temporal resolution:
- Fast movements (e.g., reaching) are coarse and need fewer key frames
- Slow movements (e.g., insertion) are delicate and need more key frames

**Advantages:**
- Dataset-driven: No manual tuning of key frame count
- Adaptive: Automatically adjusts to movement characteristics
- Efficient: Uses fewer frames for fast movements

### 7.2 6D Pose Prediction

**Rationale:** Predicting poses instead of continuous actions:
- More interpretable: Directly relates to robot's end-effector state
- Time-agnostic: No need for time alignment
- Geometrically meaningful: Loss based on actual pose distance

**Advantages:**
- No time alignment needed in loss computation
- Directly optimizes for correct end-effector poses
- Works well with sparse key frames

### 7.3 Gemma Language Model for Temporal Processing

**Rationale:** Gemma language model captures complex temporal dependencies:
- Self-attention models relationships between key frame tokens
- No fixed temporal structure (key frame count varies)
- Powerful pretrained backbone with strong sequence modeling capabilities
- Multi-modal fusion through attention between vision patches and state tokens

**Advantages:**
- Handles variable number of key frames
- Captures long-range dependencies with deep attention layers
- Leverages pretrained knowledge from large-scale text training
- Supports grouped-query attention for efficiency
- Parallelizable (unlike RNNs)

## 8. Limitations and Future Improvements

### 8.1 Current Limitations

1. **Large Model Size:** Gemma language model has ~2B parameters (or ~300M for gemma_300m), requiring significant computational resources
2. **Fixed Time Step:** Uses constant `dt=0.1s`; could be made configurable or inferred from data
3. **Pose Extraction:** Assumes actions contain pose information; may need adaptation for different action spaces
4. **No Interpolation at Inference:** Currently returns poses directly; could interpolate between key frames for smoother trajectories
5. **Vision Encoder Frozen:** SigLIP is typically frozen to save compute, limiting vision feature adaptation
6. **Memory Usage:** Processing full image sequences with Gemma can be memory-intensive for long horizons

### 8.2 Potential Improvements

1. **Vision Encoder Fine-tuning:** Enable selective fine-tuning of SigLIP layers for better task-specific adaptation
2. **Learned Key Frame Selection:** Replace heuristic with learned selection mechanism using attention
3. **Trajectory Interpolation:** Add interpolation between key frames at inference for smoother trajectories
4. **Multi-Task Learning:** Add auxiliary losses (e.g., reconstruction, segmentation)
5. **Action Space Adaptation:** Support more action space formats beyond 6D pose
6. **Quantization:** Use bfloat16 or int8 quantization to reduce memory and compute requirements
7. **Caching:** Cache vision features to avoid recomputing for repeated observations
8. **Smaller Gemma Variants:** Experiment with gemma_300m or even smaller models for resource-constrained settings
