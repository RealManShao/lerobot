export HF_USER=Xihe666
hf auth login
: "${HF_USER:?Set HF_USER to your Hugging Face username before launching training}"


# note:VLM is already frozed by default, to change it 
accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  \
  --dataset.repo_id=siemens-0816-v3 \
  --dataset.root=data/siemens-0816-v3 \
  --dataset.video_backend=torchcodec \
  \
  --policy.type=drif_ov \
  --policy.device=cuda \
  --policy.use_bf16=true \
  \
  --batch_size=32 \
  --steps=20000 \
  \
  --output_dir=outputs/train/drif_ov_siemens_0816 \
  --job_name=drif_ov_siemens_0816 \
  --policy.repo_id="${HF_USER}/drif_ov_siemens_0816" \
  \
  --wandb.enable=true

# resume version
accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --steps=20001 \
  \
  --output_dir=outputs/train/drif_ov_siemens_0817upload \
  --policy.repo_id="${HF_USER}/drif_ov_siemens_0816" \
  --resume=true \
  --config_path=outputs/train/drif_ov_siemens_0816/checkpoints/last/pretrained_model/train_config.json