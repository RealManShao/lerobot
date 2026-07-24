#!/usr/bin/env python

"""Offline full-pipeline latency benchmark for the Drifting checkpoint.

Actual command used for the 2026-07-24 measurement on this workspace:

    cd /mnt/workspace/ivanshao/lerobot
    source /mnt/workspace/ivanshao/lerobot/.venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python /mnt/workspace/ivanshao/lerobot/tests/policies/drifting/benchmark_inference_latency.py

The benchmark uses these real local paths:

    Checkpoint: /mnt/workspace/ivanshao/lerobot/outputs/train/drifting_siemens_0722/checkpoints/010002/pretrained_model
    Dataset:    /mnt/workspace/ivanshao/lerobot/data/siemens-v3-disturb

It measures three fixed frames 10 times each after warming the model, CUDA
kernels, and TorchCodec decoder cache. Model loading and Hub downloads are not
included. Run ``hf auth login`` first if the GR00T/Cosmos assets are not cached.

This is a manual GPU benchmark, intentionally not named ``test_*.py`` so that
normal pytest runs do not load the model or reserve GPU memory.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torchcodec

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION

WORKSPACE = Path("/mnt/workspace/ivanshao/lerobot")
CHECKPOINT = WORKSPACE / "outputs/train/drifting_siemens_0722/checkpoints/010002/pretrained_model"
DATASET_ROOT = WORKSPACE / "data/siemens-v3-disturb"

FRAME_INDICES = (100, 100100, 200100)
REPETITIONS = 10
CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
STATE_KEYS = (
    "observation.state",
    "observation.ee_pose_left",
    "observation.ee_pose_right",
    "observation.ee_pose_cmd_right",
    "observation.ee_pose_cmd_left",
)
TASK = "pick up a package then put it into box"
ROBOT_TYPE = "tron2"
DEVICE = torch.device("cuda")

STAGE_NAMES = (
    "video_decode_3cam",
    "observation_preparation",
    "policy_preprocessing",
    "model_input_preparation",
    "vlm_backbone",
    "drifting_action_head",
    "action_postprocessing",
    "end_to_end_action_chunk",
)


def synchronize_cuda() -> None:
    torch.cuda.synchronize()


def timed[Result](function: Callable[[], Result]) -> tuple[Result, float]:
    synchronize_cuda()
    start = perf_counter()
    result = function()
    synchronize_cuda()
    return result, (perf_counter() - start) * 1000


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(median(array)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def print_summary(name: str, values: list[float]) -> None:
    stats = summarize(values)
    print(
        f"{name:26s} n={len(values):2d} "
        f"p50={stats['p50']:8.3f} ms "
        f"p99={stats['p99']:8.3f} ms "
        f"max={stats['maximum']:8.3f} ms "
        f"mean={stats['mean']:8.3f} ms"
    )


def video_path(camera: str, episode: int) -> Path:
    return DATASET_ROOT / "videos" / camera / "chunk-000" / f"file-{episode:03d}.mp4"


def decode_observation(row: dict[str, Any], executor: ThreadPoolExecutor) -> dict[str, np.ndarray]:
    episode = int(row["episode_index"])
    timestamp = float(row["timestamp"])

    def decode_camera(camera: str) -> tuple[str, np.ndarray]:
        frames = decode_video_frames(
            video_path(camera, episode),
            timestamps=[timestamp],
            tolerance_s=1 / 30,
            backend="torchcodec",
            return_uint8=True,
        )
        image = frames[0].permute(1, 2, 0).contiguous().numpy()
        return camera, image

    observation = dict(executor.map(decode_camera, CAMERAS))
    observation.update({key: np.asarray(row[key], dtype=np.float32) for key in STATE_KEYS})
    return observation


def prepare_observation(observation: dict[str, np.ndarray]) -> dict[str, Any]:
    copied = {key: value.copy() for key, value in observation.items()}
    return prepare_observation_for_inference(copied, DEVICE, TASK, ROBOT_TYPE)


def main() -> None:
    if not CHECKPOINT.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATASET_ROOT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This benchmark requires an NVIDIA GPU.")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"torch={torch.__version__}")
    print(f"torchcodec={torchcodec.__version__}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"checkpoint={CHECKPOINT}")
    print(f"dataset={DATASET_ROOT}")

    dataset = LeRobotDataset(
        repo_id="siemens-v3-disturb",
        root=DATASET_ROOT,
        video_backend="torchcodec",
        return_uint8=True,
    )
    rows = [dataset.get_raw_item(index) for index in FRAME_INDICES]
    for index, row in zip(FRAME_INDICES, rows, strict=True):
        print(
            f"sample index={index} episode={int(row['episode_index'])} "
            f"timestamp={float(row['timestamp']):.3f}s"
        )

    config = PreTrainedConfig.from_pretrained(CHECKPOINT)
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(CHECKPOINT, config=config).eval()
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(CHECKPOINT))

    def prepare_model_inputs(processed: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        inputs = policy._filter_groot_inputs(processed, include_action=False)
        return policy._groot_model.prepare_input(inputs)

    def run_backbone(backbone_inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config.use_bf16):
            return policy._groot_model.backbone(backbone_inputs)

    def run_action_head(
        backbone_output: dict[str, torch.Tensor], action_inputs: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config.use_bf16):
            return policy._groot_model.action_head.get_action(backbone_output, action_inputs)

    stage_times = {name: [] for name in STAGE_NAMES}

    with ThreadPoolExecutor(max_workers=len(CAMERAS)) as executor:
        # Warm every fixed episode's decoder entries and both model paths.
        warm_observations = [decode_observation(row, executor) for row in rows]
        for _ in range(2):
            for observation in warm_observations:
                prepared = prepare_observation(observation)
                processed = preprocessor(prepared)
                backbone_inputs, action_inputs = prepare_model_inputs(processed)
                backbone_output = run_backbone(backbone_inputs)
                run_action_head(backbone_output, action_inputs)
        synchronize_cuda()

        for _ in range(REPETITIONS):
            for row in rows:
                synchronize_cuda()
                end_to_end_start = perf_counter()

                observation, elapsed = timed(lambda row=row: decode_observation(row, executor))
                stage_times["video_decode_3cam"].append(elapsed)

                prepared, elapsed = timed(lambda observation=observation: prepare_observation(observation))
                stage_times["observation_preparation"].append(elapsed)

                processed, elapsed = timed(lambda prepared=prepared: preprocessor(prepared))
                stage_times["policy_preprocessing"].append(elapsed)

                model_inputs, elapsed = timed(lambda processed=processed: prepare_model_inputs(processed))
                stage_times["model_input_preparation"].append(elapsed)
                backbone_inputs, action_inputs = model_inputs

                backbone_output, elapsed = timed(
                    lambda backbone_inputs=backbone_inputs: run_backbone(backbone_inputs)
                )
                stage_times["vlm_backbone"].append(elapsed)

                head_output, elapsed = timed(
                    lambda backbone_output=backbone_output, action_inputs=action_inputs: run_action_head(
                        backbone_output, action_inputs
                    )
                )
                stage_times["drifting_action_head"].append(elapsed)

                action_dim = config.output_features[ACTION].shape[0]
                action = head_output["action_pred"][:, 0, :action_dim]
                _, elapsed = timed(lambda action=action: postprocessor(action))
                stage_times["action_postprocessing"].append(elapsed)

                synchronize_cuda()
                stage_times["end_to_end_action_chunk"].append((perf_counter() - end_to_end_start) * 1000)

        queue_observation = decode_observation(rows[0], executor)

    queue_input = preprocessor(prepare_observation(queue_observation))
    policy.reset()
    _, first_refill_ms = timed(lambda: policy.select_action(queue_input))
    queue_steps = int(policy._action_queue_steps)
    queue_hit_times = []
    for _ in range(queue_steps - 1):
        _, elapsed = timed(lambda: policy.select_action(queue_input))
        queue_hit_times.append(elapsed)
    _, next_refill_ms = timed(lambda: policy.select_action(queue_input))

    print("\nStage results (warm cache)")
    for name in STAGE_NAMES:
        print_summary(name, stage_times[name])

    print("\nSynchronous action queue")
    print(f"first chunk refill         {first_refill_ms:8.3f} ms")
    print_summary(f"queue hits ({queue_steps - 1})", queue_hit_times)
    print(f"next chunk refill          {next_refill_ms:8.3f} ms")

    print("\nCUDA memory")
    print(f"peak allocated={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    print(f"peak reserved={torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
