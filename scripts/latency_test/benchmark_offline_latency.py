#!/usr/bin/env python
"""Offline full-pipeline inference latency benchmark for Drifting-family policies.

Measures per-stage GPU latency (video decode → observation prep → preprocessor →
model input → VLM backbone → action head → postprocessor) over warm-cache repeated
runs, then reports P50 / P99 / Max / Mean for each stage and the synchronous action
queue behaviour.

This is a manual GPU benchmark — intentionally not named ``test_*.py`` so that
normal pytest runs never load a model or reserve GPU memory.

Supported policies (select with --policy):
  drifting    Xihe666/drifting_libero_full or a local path          (default)
  drif_ov     Xihe666/drif_ov_libero0809   or a local path
  groot_n17   Xihe666/gr00t_n17_libero_1A800_0811  or a local path

Usage
-----
    # Run with built-in defaults for one policy:
    CUDA_VISIBLE_DEVICES=0 python scripts/latency_test/benchmark_offline_latency.py --policy drifting
    CUDA_VISIBLE_DEVICES=0 python scripts/latency_test/benchmark_offline_latency.py --policy drif_ov
    CUDA_VISIBLE_DEVICES=0 python scripts/latency_test/benchmark_offline_latency.py --policy groot_n17

    # Override checkpoint and/or dataset:
    CUDA_VISIBLE_DEVICES=0 python scripts/latency_test/benchmark_offline_latency.py \\
        --policy drifting \\
        --checkpoint outputs/train/drifting_siemens_0722/checkpoints/010002/pretrained_model \\
        --dataset data/siemens-v3-disturb

    # Adjust repetitions:
    CUDA_VISIBLE_DEVICES=0 python scripts/latency_test/benchmark_offline_latency.py \\
        --policy drif_ov --reps 20

Run ``hf auth login`` first if the checkpoint is not yet cached locally.
Model loading, Hub downloads, and first-kernel JIT compilation are NOT included
in any reported times.
"""

from __future__ import annotations

import argparse
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

# ── Per-policy defaults (override via CLI) ────────────────────────────────────
_POLICY_DEFAULTS: dict[str, dict] = {
    "drifting": {
        "checkpoint": "outputs/train/drifting_siemens_0722/checkpoints/010002/pretrained_model",
        "dataset_root": "data/siemens-v3-disturb",
        "dataset_repo_id": "siemens-v3-disturb",
        "cameras": (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ),
        "state_keys": (
            "observation.state",
            "observation.ee_pose_left",
            "observation.ee_pose_right",
            "observation.ee_pose_cmd_right",
            "observation.ee_pose_cmd_left",
        ),
        "task": "pick up a package then put it into box",
        "robot_type": "tron2",
        "frame_indices": (100, 100100, 200100),
        "action_head_stage_name": "drifting_action_head",
        "video_decode_stage_name": "video_decode_3cam",
    },
    "drif_ov": {
        "checkpoint": "Xihe666/drif_ov_libero0809",
        "dataset_root": "data/libero-10",
        "dataset_repo_id": "libero-10",
        "cameras": (
            "observation.images.agentview_rgb",
            "observation.images.eye_in_hand_rgb",
        ),
        "state_keys": ("observation.state",),
        "task": "pick up the alphabet soup and place it in the basket",
        "robot_type": "panda",
        "frame_indices": (100, 10100, 20100),
        "action_head_stage_name": "drif_ov_action_head",
        "video_decode_stage_name": "video_decode_2cam",
    },
    "groot_n17": {
        "checkpoint": "Xihe666/gr00t_n17_libero_1A800_0811",
        "dataset_root": "data/libero-10",
        "dataset_repo_id": "libero-10",
        "cameras": (
            "observation.images.agentview_rgb",
            "observation.images.eye_in_hand_rgb",
        ),
        "state_keys": ("observation.state",),
        "task": "pick up the alphabet soup and place it in the basket",
        "robot_type": "panda",
        "frame_indices": (100, 10100, 20100),
        "action_head_stage_name": "groot_action_head",
        "video_decode_stage_name": "video_decode_2cam",
    },
}

DEVICE = torch.device("cuda")


# ── Timing helpers ────────────────────────────────────────────────────────────

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
    s = summarize(values)
    print(
        f"{name:28s} n={len(values):2d} "
        f"p50={s['p50']:8.3f} ms  p99={s['p99']:8.3f} ms  "
        f"max={s['maximum']:8.3f} ms  mean={s['mean']:8.3f} ms"
    )


# ── Dataset helpers ───────────────────────────────────────────────────────────

def video_path(dataset_root: Path, camera: str, episode: int) -> Path:
    return dataset_root / "videos" / camera / "chunk-000" / f"file-{episode:03d}.mp4"


def decode_observation(
    row: dict[str, Any],
    executor: ThreadPoolExecutor,
    *,
    dataset_root: Path,
    cameras: tuple[str, ...],
    state_keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    episode = int(row["episode_index"])
    timestamp = float(row["timestamp"])

    def decode_camera(camera: str) -> tuple[str, np.ndarray]:
        frames = decode_video_frames(
            video_path(dataset_root, camera, episode),
            timestamps=[timestamp],
            tolerance_s=1 / 30,
            backend="torchcodec",
            return_uint8=True,
        )
        return camera, frames[0].permute(1, 2, 0).contiguous().numpy()

    observation = dict(executor.map(decode_camera, cameras))
    observation.update({key: np.asarray(row[key], dtype=np.float32) for key in state_keys})
    return observation


def prepare_observation(
    observation: dict[str, np.ndarray],
    *,
    task: str,
    robot_type: str,
) -> dict[str, Any]:
    copied = {key: value.copy() for key, value in observation.items()}
    return prepare_observation_for_inference(copied, DEVICE, task, robot_type)


# ── Main benchmark ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--policy",
        choices=list(_POLICY_DEFAULTS),
        default="drifting",
        help="Policy preset to benchmark (default: drifting).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Local path or HF repo-id for the checkpoint (overrides preset default).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        metavar="PATH",
        help="Local dataset root directory (overrides preset default).",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=10,
        metavar="N",
        help="Repetitions per frame (default: 10; total samples = reps × 3 frames).",
    )
    args = parser.parse_args()

    cfg = dict(_POLICY_DEFAULTS[args.policy])
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(cfg["checkpoint"])
    dataset_root = Path(args.dataset) if args.dataset else Path(cfg["dataset_root"])
    cameras: tuple[str, ...] = cfg["cameras"]
    state_keys: tuple[str, ...] = cfg["state_keys"]
    task: str = cfg["task"]
    robot_type: str = cfg["robot_type"]
    frame_indices: tuple[int, ...] = cfg["frame_indices"]
    action_head_stage: str = cfg["action_head_stage_name"]
    video_decode_stage: str = cfg["video_decode_stage_name"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This benchmark requires an NVIDIA GPU.")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"policy={args.policy}")
    print(f"torch={torch.__version__}  torchcodec={torchcodec.__version__}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"checkpoint={checkpoint}")
    print(f"dataset={dataset_root}")

    stage_names = (
        video_decode_stage,
        "observation_preparation",
        "policy_preprocessing",
        "model_input_preparation",
        "vlm_backbone",
        action_head_stage,
        "action_postprocessing",
        "end_to_end_action_chunk",
    )

    dataset = LeRobotDataset(
        repo_id=cfg["dataset_repo_id"],
        root=dataset_root,
        video_backend="torchcodec",
        return_uint8=True,
    )
    rows = [dataset.get_raw_item(i) for i in frame_indices]
    for idx, row in zip(frame_indices, rows, strict=True):
        print(
            f"  sample idx={idx}  episode={int(row['episode_index'])}  "
            f"timestamp={float(row['timestamp']):.3f}s"
        )

    policy_config = PreTrainedConfig.from_pretrained(str(checkpoint))
    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(str(checkpoint), config=policy_config).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_config, pretrained_path=str(checkpoint)
    )

    def prepare_model_inputs(processed: dict[str, Any]) -> tuple[Any, Any]:
        inputs = policy._filter_groot_inputs(processed, include_action=False)
        return policy._groot_model.prepare_input(inputs)

    def run_backbone(backbone_inputs: Any) -> Any:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=policy_config.use_bf16):
            return policy._groot_model.backbone(backbone_inputs)

    def run_action_head(backbone_output: Any, action_inputs: Any) -> Any:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=policy_config.use_bf16):
            return policy._groot_model.action_head.get_action(backbone_output, action_inputs)

    _decode = lambda row, ex: decode_observation(
        row, ex, dataset_root=dataset_root, cameras=cameras, state_keys=state_keys
    )
    _prepare = lambda obs: prepare_observation(obs, task=task, robot_type=robot_type)

    stage_times: dict[str, list[float]] = {name: [] for name in stage_names}

    with ThreadPoolExecutor(max_workers=len(cameras)) as executor:
        # Warm model, CUDA kernels, and decoder cache.
        warm_obs = [_decode(row, executor) for row in rows]
        for _ in range(2):
            for obs in warm_obs:
                processed = preprocessor(_prepare(obs))
                bi, ai = prepare_model_inputs(processed)
                run_action_head(run_backbone(bi), ai)
        synchronize_cuda()

        for _ in range(args.reps):
            for row in rows:
                synchronize_cuda()
                e2e_start = perf_counter()

                obs, elapsed = timed(lambda r=row: _decode(r, executor))
                stage_times[video_decode_stage].append(elapsed)

                prepared, elapsed = timed(lambda o=obs: _prepare(o))
                stage_times["observation_preparation"].append(elapsed)

                processed, elapsed = timed(lambda p=prepared: preprocessor(p))
                stage_times["policy_preprocessing"].append(elapsed)

                model_inputs, elapsed = timed(lambda p=processed: prepare_model_inputs(p))
                stage_times["model_input_preparation"].append(elapsed)
                backbone_inputs, action_inputs = model_inputs

                backbone_output, elapsed = timed(lambda bi=backbone_inputs: run_backbone(bi))
                stage_times["vlm_backbone"].append(elapsed)

                head_output, elapsed = timed(
                    lambda bo=backbone_output, ai=action_inputs: run_action_head(bo, ai)
                )
                stage_times[action_head_stage].append(elapsed)

                action_dim = policy_config.output_features[ACTION].shape[0]
                action = head_output["action_pred"][:, 0, :action_dim]
                _, elapsed = timed(lambda a=action: postprocessor(a))
                stage_times["action_postprocessing"].append(elapsed)

                synchronize_cuda()
                stage_times["end_to_end_action_chunk"].append(
                    (perf_counter() - e2e_start) * 1000
                )

        queue_obs = _decode(rows[0], executor)

    queue_input = preprocessor(_prepare(queue_obs))
    policy.reset()
    _, first_refill_ms = timed(lambda: policy.select_action(queue_input))
    queue_steps = int(policy._action_queue_steps)
    queue_hit_times: list[float] = []
    for _ in range(queue_steps - 1):
        _, elapsed = timed(lambda: policy.select_action(queue_input))
        queue_hit_times.append(elapsed)
    _, next_refill_ms = timed(lambda: policy.select_action(queue_input))

    print(f"\nStage results — warm cache  (reps={args.reps}, frames=3, n={args.reps * 3})")
    for name in stage_names:
        print_summary(name, stage_times[name])

    print("\nSynchronous action queue")
    print(f"  first chunk refill       {first_refill_ms:8.3f} ms")
    if queue_hit_times:
        print_summary(f"  queue hits ({queue_steps - 1})", queue_hit_times)
    print(f"  next chunk refill        {next_refill_ms:8.3f} ms")

    print("\nCUDA memory (peak)")
    print(f"  allocated={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB  "
          f"reserved={torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
