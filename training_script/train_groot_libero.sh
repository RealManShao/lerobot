# Train gr00t n17 based on LIBERO dataset
# Unified training setting: batch size=32, steps=20k, use bf16, 2*A800 80 GB
# Need change: seed=1, 42, 1000(default if blank),

accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --mixed_precision=bf16 \
  --num_machines=1 \
  --dynamo_backend=no \
  $(which lerobot-train) \
  --dataset.repo_id=lerobot/libero \
  --dataset.video_backend=torchcodec \
  --policy.type=groot \
  --policy.device=cuda \
  --policy.use_bf16=true \
  --policy.push_to_hub=true \
  --policy.repo_id="Xihe666/gr00t_n17_libero_20k_seed1" \
  --batch_size=32 \
  --steps=20000 \
  --save_checkpoint_to_hub=true \
  --output_dir=outputs/train/libero/groot/0821/try1-bs32x2-20000 \
  --job_name=gr00t_n17_libero-2gpu-bs32x2-0821-seed1 \
  --seed=1 \
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