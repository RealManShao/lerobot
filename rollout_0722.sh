#!/usr/bin/env bash
set -euo pipefail

cd /mnt/workspace/ivanshao/lerobot
source /root/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
export TRON2_ROBOT_IP=...
export TRON2_CAM_HIGH_SERIAL=...
export TRON2_CAM_LEFT_WRIST_SERIAL=...
export TRON2_CAM_RIGHT_WRIST_SERIAL=...
CUDA_VISIBLE_DEVICES=0 lerobot-rollout \
  --strategy.type=base \
  --inference.type=sync \
  --policy.path=outputs/train/drifting_siemens_0722/checkpoints/010002/pretrained_model \
  --robot.type=tron2 \
  --robot.robot_ip="${TRON2_ROBOT_IP:?Set TRON2_ROBOT_IP}" \
  --robot.cameras="{cam_high: {type: intelrealsense, serial_number_or_name: ${TRON2_CAM_HIGH_SERIAL:?Set TRON2_CAM_HIGH_SERIAL}, width: 640, height: 480, fps: 30}, cam_left_wrist: {type: intelrealsense, serial_number_or_name: ${TRON2_CAM_LEFT_WRIST_SERIAL:?Set TRON2_CAM_LEFT_WRIST_SERIAL}, width: 640, height: 480, fps: 30}, cam_right_wrist: {type: intelrealsense, serial_number_or_name: ${TRON2_CAM_RIGHT_WRIST_SERIAL:?Set TRON2_CAM_RIGHT_WRIST_SERIAL}, width: 640, height: 480, fps: 30}}" \
  --fps=30 \
  --device=cuda \
  --task="pick up a package then put it into box" \
  --duration=120
