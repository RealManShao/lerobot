# Train Drifting based on LIBERO dataset
export MUJOCO_GL=egl
export OMP_NUM_THREADS=6
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lerobot/libero \
  --dataset.video_backend=torchcodec \
  --policy.type=drif_ov \
  --batch_size=256 \
  --steps=15000 \
  --output_dir=outputs/train/libero/drif_ov/try1 \
  --job_name=drif_ov_libero \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id="Xihe666/drif_ov_libero0809"

# Finetune drifting on LIBERO 10
IMAGE_TRANSFORMS='{
  "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.7, 1.3]}},
  "contrast":   {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast":   [0.6, 1.4]}},
  "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
  "hue":        {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue":        [-0.08, 0.08]}}
}'

  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=tailong-wu/libero_10_no_noops_1.0.0_lerobot_v3.0 \
  --dataset.video_backend=torchcodec \
    --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=4 \
  --dataset.image_transforms.tfs="$IMAGE_TRANSFORMS" \
  --policy.type=drifting \
  --policy.base_model_path=Xihe666/drifting_libero_full \
  --batch_size=400 \
  --steps=10000 \
  --output_dir=outputs/train/libero/finetune/libero_10/try4 \
  --job_name=drifting_libero_10 \
  --policy.device=cuda \
  --wandb.enable=true \
  --persistent_workers=true \
  --num_workers=16 \
  --prefetch_factor=2 \
  --policy.repo_id="Xihe666/drifting_libero_10"