# My Custom Policy for LeRobot

A custom VLA (Vision-Language-Action) policy implementation for LeRobot with four key innovations:

1. **SigLIP Vision Encoder**: Uses the SigLIP vision encoder from PaliGemma for powerful visual feature extraction, with a multi-modal projector to align vision features with the language model space.
2. **Gemma Language Model**: Employs Gemma (gemma_2b or gemma_300m) as the temporal encoder, leveraging pretrained language model capabilities for complex sequence modeling.
3. **Speed-Based Key Frame Selection**: Automatically selects key frames based on end-effector movement speed. High-speed movements use fewer key frames (accuracy less critical), while low-speed movements use more key frames for precision. The number of key frames is determined by the dataset, not user configuration.
4. **6D Pose Prediction**: Predicts desired end-effector 6D pose (position + orientation) and gripper value at key frames. Loss is computed as the distance between predicted pose and ground truth trajectory, with no time alignment requirement.
5. **Negative Sample Learning**: Learns from both successful and near-failure attempts by manually labeling failure points, allowing the model to pay more attention to critical steps for success.

## Features

### Speed-Based Key Frame Selection

The policy automatically determines the number and spacing of key frames based on the end-effector's average speed:

- **High speed movement**: Fewer key frames (movement is fast, accuracy less critical)
- **Low speed movement**: More key frames (movement is slow, need more precision)
- **Time window**: Ensures minimum spacing between key frames
- **Dataset-driven**: Key frames are determined by the actual movement in the data, not user configuration

Key frames are the only input to the model - the model processes observations at key frame indices and predicts poses for those frames.

### 6D Pose Prediction

Instead of predicting continuous actions, the model predicts 6D pose (position + quaternion) + gripper at each key frame:

- **Position**: 3D coordinates (x, y, z)
- **Orientation**: Quaternion (qx, qy, qz, qw)
- **Gripper**: Binary or continuous gripper value

The loss function computes:
- Position distance (L2 loss)
- Orientation distance (quaternion angular distance)
- Gripper loss (MSE)

No time alignment is needed - the model just predicts poses at key frames.

### Negative Sample Learning

The policy can learn from near-failure demonstrations where the task almost succeeded but failed at a critical step (e.g., catching a stick and moving to a hole but failing insertion). This is achieved by:

- **Manual failure labeling**: You label the timestep where failure occurred
- **Focal loss weighting**: Applies higher loss weights around failure points using focal loss
- **Attention to key steps**: The model learns to focus on the critical moments that determine success or failure

## Installation

```bash
cd lerobot_policy_my_custom_policy
pip install -e .
```

## Configuration Parameters

### Vision Encoder Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vision_encoder_name` | str | "google/paligemma-3b-pt-224" | HuggingFace model name for vision encoder |
| `image_resolution` | tuple | (224, 224) | Image resolution (height, width) |
| `freeze_vision_encoder` | bool | False | Whether to freeze vision encoder |

### Gemma Language Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gemma_variant` | str | "gemma_2b" | Gemma variant ("gemma_2b" or "gemma_300m") |
| `gemma_hidden_size` | int | 2048 | Hidden size for Gemma |
| `gemma_num_layers` | int | 18 | Number of layers in Gemma |
| `gemma_num_heads` | int | 8 | Number of attention heads |
| `gemma_num_kv_heads` | int | 1 | Number of key-value heads |
| `gemma_head_dim` | int | 256 | Head dimension |
| `gemma_mlp_dim` | int | 16384 | MLP dimension |
| `dtype` | str | "float32" | Data type ("float32" or "bfloat16") |

### Speed-Based Key Frame Selection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_speed_threshold` | float | 0.01 | Minimum speed to consider movement |
| `max_speed_threshold` | float | 2.0 | Maximum speed for dense key frames |
| `time_window` | float | 0.5 | Time window in seconds for key frame spacing |
| `min_key_frames` | int | 2 | Minimum number of key frames (start and end) |
| `max_key_frames` | int | 20 | Maximum number of key frames to prevent over-segmentation |

### Negative Sample Learning Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_negative_samples` | bool | True | Enable learning from negative samples |
| `negative_sample_weight` | float | 2.0 | Weight multiplier for failure point loss |
| `failure_label_key` | str | "failure_point" | Key in batch for failure timestep indices |
| `focal_loss_gamma` | float | 2.0 | Gamma parameter for focal loss |

### Standard Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `horizon` | int | 50 | Total action prediction horizon |
| `n_action_steps` | int | 50 | Number of action steps to execute at inference |
| `hidden_dim` | int | 256 | Deprecated - use `gemma_hidden_size` instead |
| `optimizer_lr` | float | 1e-4 | Learning rate |
| `optimizer_weight_decay` | float | 1e-4 | Weight decay for optimizer |

## Usage

### Basic Training

The policy automatically selects key frames based on end-effector speed from your dataset. No manual key frame configuration is needed.

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=my_custom_policy \
  --dataset.repo_id=your_org/your_dataset
```

### Training with Custom Speed Thresholds

Adjust speed-based key frame selection parameters for your specific task:

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=my_custom_policy \
  --dataset.repo_id=your_org/your_dataset \
  --policy.min_speed_threshold=0.02 \
  --policy.max_speed_threshold=1.5 \
  --policy.time_window=0.3 \
  --policy.min_key_frames=3 \
  --policy.max_key_frames=15
```

### Training with Negative Sample Learning

First, prepare your dataset with failure point labels. Add a `failure_point` field to your dataset metadata containing the timestep index where failure occurred (-1 for successful episodes).

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=my_custom_policy \
  --dataset.repo_id=your_org/your_dataset \
  --policy.use_negative_samples=true \
  --policy.negative_sample_weight=2.0 \
  --policy.focal_loss_gamma=2.0
```

### Training with Both Features

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=my_custom_policy \
  --dataset.repo_id=your_org/your_dataset \
  --policy.min_speed_threshold=0.01 \
  --policy.max_speed_threshold=2.0 \
  --policy.time_window=0.5 \
  --policy.min_key_frames=2 \
  --policy.max_key_frames=20 \
  --policy.use_negative_samples=true \
  --policy.negative_sample_weight=3.0 \
  --policy.focal_loss_gamma=2.0
```

## Dataset Preparation

### For Key Frame Training

No special preparation is needed for key frame training. The policy automatically extracts key frames from your existing data.

### For Negative Sample Learning

You need to add failure point labels to your dataset. For each episode in your dataset:

1. **Successful episodes**: Set `failure_point = -1`
2. **Failed episodes**: Set `failure_point = <timestep_index>` where the failure occurred

Example dataset structure:

```python
{
    "episode": 0,
    "observation": {
        "image": [...],  # Your image data
        "state": [...],  # Your state data
    },
    "action": [...],  # Your action data
    "failure_point": 25,  # Failure occurred at timestep 25
}
```

For plug-in-hole task example:
- Successfully catch the stick and move to hole, but fail insertion: `failure_point = 40` (insertion step)
- Complete success: `failure_point = -1`

## Model Architecture

### KeyFrameEncoder

- **Vision Encoder**: SigLIP vision encoder from PaliGemma with multi-modal projector to align features with Gemma space
- **State Encoder**: Linear layer projecting state to gemma_hidden_size
- **Feature Concatenation**: Combines vision patches and state token for each key frame
- **Gemma Language Model**: Gemma (gemma_2b or gemma_300m) processes the concatenated sequence for temporal understanding
- **Feature Extraction**: Extracts state token features from Gemma output for each key frame
- **Pose Head**: Predicts 6D pose (position + quaternion) and gripper at key frames

### Negative Sample Learning

The model applies focal loss weighting:
- Higher weight for timesteps around failure points
- Weight determined by prediction error: `(1 - pt)^gamma`
- Key frames also receive additional weighting

## Example: Plug-in-Hole Task

### Scenario
You have demonstrations where the robot:
1. Successfully catches the stick
2. Moves to the hole
3. Fails to insert (gets stuck at the hole entrance)

### Labeling
Label the failure point as the timestep where insertion fails:

```python
failure_point = 42  # Assuming timestep 42 is where insertion fails
```

### Training
```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=my_custom_policy \
  --policy.repo_id=${HF_USER}/metaworld-test \
  --dataset.repo_id=lerobot/metaworld_mt50 \
  --env.type=metaworld \
  --env.task=assembly-v3,dial-turn-v3,handle-press-side-v3 \
  --output_dir=./outputs/ \
  --policy.min_speed_threshold=0.01 \
  --policy.max_speed_threshold=2.0 \
  --policy.time_window=2 \
  --policy.min_key_frames=2 \
  --policy.max_key_frames=20 \
  --policy.use_negative_samples=false \
  --policy.negative_sample_weight=3.0 \
  --policy.focal_loss_gamma=2.0
```

The model will learn to:
- Automatically select key frames based on movement speed during insertion
- Pay special attention to the insertion step where failures occur
- Predict 6D poses (position + orientation) at critical decision points

## Advanced Usage

### Adjusting Speed Thresholds

For tasks with specific speed characteristics, adjust the speed thresholds:

```python
# In your training script
policy_config.min_speed_threshold = 0.02  # Higher threshold for fast movements
policy_config.max_speed_threshold = 1.0    # Lower threshold for more key frames
policy_config.time_window = 0.3             # Smaller window for denser key frames
```

### Adjusting Focal Loss

Increase gamma to focus more on hard examples:

```python
policy_config.focal_loss_gamma = 3.0  # More aggressive focusing
```

### Adjusting Key Frame Bounds

Control the range of key frames the model can select:

```python
policy_config.min_key_frames = 3   # Ensure at least 3 key frames
policy_config.max_key_frames = 15  # Limit to prevent over-segmentation
```

## Implementation Details

### Speed-Based Key Frame Selection

- **Compute EEF speed**: Calculates end-effector speed from pose trajectory
- **Determine key frame count**: Based on average speed (high speed → fewer frames, low speed → more frames)
- **Apply time window**: Ensures minimum spacing between key frames
- **Always includes**: First and last timesteps are always key frames

### Pose Prediction

The model predicts 6D pose (position + quaternion) + gripper at key frames:
- **Position**: 3D coordinates (x, y, z)
- **Orientation**: Quaternion (qx, qy, qz, qw)
- **Gripper**: Binary or continuous value

### Loss Computation

The total loss combines:
1. Position distance loss (L2)
2. Orientation distance loss (quaternion angular distance)
3. Gripper loss (MSE)
4. Weighted loss at failure points (negative samples)

```
loss = pos_loss + quat_loss + gripper_loss
# Apply focal loss weighting for failure points
sample_weights[b, t] = 1.0 + failure_weight[b, t]
```

## Troubleshooting

### Issue: Model not learning from failure points
- **Solution**: Increase `negative_sample_weight` or `focal_loss_gamma`
- **Check**: Verify failure_point labels are correctly set in your dataset

### Issue: Key frames not improving performance
- **Solution**: Adjust `min_speed_threshold` and `max_speed_threshold` for your task's speed characteristics
- **Solution**: Try different `time_window` values to change key frame spacing
- **Solution**: Adjust `min_key_frames` and `max_key_frames` bounds

### Issue: Training too slow
- **Solution**: Reduce `hidden_dim` for a smaller model
- **Solution**: Reduce `max_key_frames` to limit the number of key frames processed

### Issue: Poor pose prediction accuracy
- **Solution**: Ensure your action space contains 6D pose (position + quaternion) + gripper
- **Check**: Verify pose data is properly normalized in your dataset

## Citation

If you use this policy in your research, please cite:

```bibtex
@software{my_custom_policy,
  title={My Custom Policy for LeRobot},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/lerobot_policy_my_custom_policy}
}
```

## License

This policy follows the same license as LeRobot (Apache 2.0).
