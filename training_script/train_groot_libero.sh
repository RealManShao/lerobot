# Train gr00t n17 based on LIBERO dataset
export MUJOCO_GL=egl
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lerobot/libero \
  --dataset.video_backend=torchcodec \
  --policy.type=groot \
  --policy.device=cuda \
  --policy.use_bf16=true \
  --policy.push_to_hub=true \
  --policy.repo_id="Xihe666/gr00t_n17_libero" \
  --batch_size=128 \
  --steps=20000 \
  --output_dir=outputs/train/libero/groot/0810/try1-bs128 \
  --job_name=gr00t_n17_libero \
  --wandb.enable=true

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lerobot/libero \
  --dataset.video_backend=torchcodec \
  --policy.type=groot \
  --policy.device=cuda \
  --policy.use_bf16=true \
  --policy.push_to_hub=true \
  --policy.repo_id="Xihe666/gr00t_n17_libero" \
  --batch_size=128 \
  --steps=10000 \
  --save_freq=5000 \
  --save_checkpoint_to_hub=true \
  --output_dir=outputs/train/libero/groot/0810/try2-bs128 \
  --job_name=gr00t_n17_libero \
  --wandb.enable=true