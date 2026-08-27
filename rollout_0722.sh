#!/usr/bin/env bash
# set -euo pipefail

# cd /mnt/workspace/ivanshao/lerobot
# source /root/miniforge3/etc/profile.d/conda.sh
# conda activate lerobot

# use by 
# uv run bash rollout_0722.sh

export TRON2_ROBOT_IP="${TRON2_ROBOT_IP:-10.192.1.2}"
export TRON2_BRIDGE_HOST="${TRON2_BRIDGE_HOST:-wss://10.192.1.4}"
export CUDA_VISIBLE_DEVICES=0
lerobot-rollout \
  --strategy.type=base \
  --inference.type=drifov_overlap \
  --policy.n_action_steps=16 \
  --interpolation_multiplier=2 \
  --policy.path=Xihe666/drif_ov_siemens_0816 \
  --robot.type=tron2 \
  --robot.robot_ip="${TRON2_ROBOT_IP:?Set TRON2_ROBOT_IP}" \
  --robot.observation_source=bridge \
  --robot.bridge_host="${TRON2_BRIDGE_HOST:?Set TRON2_BRIDGE_HOST}" \
  --robot.bridge_camera_width=640 \
  --robot.bridge_camera_height=480 \
  --fps=15 \
  --device=cuda \
  --task="pick up a package then put it into box" \
  --duration=60


# benchmark libero policies for pi05
lerobot-eval \
  --policy.path=Xihe666/pi05_libero_full_2A800_bs32_20k \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=2 \
  --eval.n_episodes=3

  # libero_spatial
uv run python scripts/latency_test/benchmark_libero_latency.py \
    --policies Xihe666/pi05_libero_full_2A800_bs32_20k \
    --task libero_spatial \
    --n-episodes 5 \
    --output-dir outputs/latency_bench

# libero_object
uv run python scripts/latency_test/benchmark_libero_latency.py \
    --policies Xihe666/pi05_libero_full_2A800_bs32_20k \
    --task libero_object \
    --n-episodes 5 \
    --output-dir outputs/latency_bench

# libero_goal
uv run python scripts/latency_test/benchmark_libero_latency.py \
    --policies Xihe666/pi05_libero_full_2A800_bs32_20k \
    --task libero_goal \
    --n-episodes 5 \
    --output-dir outputs/latency_bench

# libero_10
uv run python scripts/latency_test/benchmark_libero_latency.py \
    --policies Xihe666/pi05_libero_full_2A800_bs32_20k \
    --task libero_10 \
    --n-episodes 5 \
    --output-dir outputs/latency_bench