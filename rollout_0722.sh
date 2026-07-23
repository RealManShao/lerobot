#!/usr/bin/env bash
# set -euo pipefail

# cd /mnt/workspace/ivanshao/lerobot
# source /root/miniforge3/etc/profile.d/conda.sh
# conda activate lerobot
export TRON2_ROBOT_IP="${TRON2_ROBOT_IP:-10.192.1.2}"
export TRON2_BRIDGE_HOST="${TRON2_BRIDGE_HOST:-wss://10.192.1.4}"
CUDA_VISIBLE_DEVICES=0 lerobot-rollout \
  --strategy.type=base \
  --inference.type=rtc \
  --inference.rtc.enabled=false \
  --inference.queue_threshold=5 \
  --policy.path=Xihe666/drifting_siemens_0722 \
  --robot.type=tron2 \
  --robot.robot_ip="${TRON2_ROBOT_IP:?Set TRON2_ROBOT_IP}" \
  --robot.observation_source=bridge \
  --robot.bridge_host="${TRON2_BRIDGE_HOST:?Set TRON2_BRIDGE_HOST}" \
  --robot.bridge_camera_width=640 \
  --robot.bridge_camera_height=480 \
  --fps=30 \
  --device=cuda \
  --task="pick up a package then put it into box" \
  --duration=60
