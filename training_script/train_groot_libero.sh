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

accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=lerobot/libero \
  --dataset.video_backend=torchcodec \
  --policy.type=groot \
  --policy.device=cuda \
  --policy.use_bf16=true \
  --policy.push_to_hub=true \
  --policy.repo_id="Xihe666/gr00t_n17_libero" \
  --batch_size=32 \
  --steps=15001 \
  --save_freq=5000 \
  --save_checkpoint_to_hub=true \
  --output_dir=outputs/train/libero/groot/0813/try1-bs32x2-15000 \
  --job_name=gr00t_n17_libero-2gpu-bs32x2-0813 \
  --wandb.enable=true 

accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --policy.type=groot \
  --policy.repo_id="Xihe666/gr00t_n17_libero" \
  --steps=20001 \
  --save_freq=5000 \
  --output_dir=outputs/train/libero/groot/0813/try1-bs32x2-20000 \
  --job_name=gr00t_n17_libero-4gpu-bs32x2-20000 \
  --resume=true \
  --config_path=outputs/train/libero/groot/0813/try1-bs32x2-15000/checkpoints/last/pretrained_model/train_config.json