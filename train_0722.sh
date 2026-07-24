# Completed training run (checkpoint saved at step 10000).
# cd /mnt/workspace/ivanshao/lerobot
# source /root/miniforge3/etc/profile.d/conda.sh
# conda activate lerobot
# CUDA_VISIBLE_DEVICES=0,1,2 /usr/local/bin/torchrun --standalone --nproc_per_node=3 -m lerobot.scripts.lerobot_train \
#   --dataset.repo_id=siemens-v3-disturb \
#   --dataset.root=/mnt/workspace/ivanshao/lerobot/data/siemens-v3-disturb \
#   --dataset.video_backend=torchcodec \
#   --policy.type=drifting \
#   --batch_size=96 \
#   --steps=10000 \
#   --output_dir=outputs/train/drifting_siemens_0722 \
#   --job_name=drifting_siemens_0722 \
#   --policy.device=cuda \
#   --wandb.enable=true \
#   --policy.repo_id="${HF_USER}/drifting_siemens_0722"

cd /mnt/workspace/ivanshao/lerobot
source /root/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
export HF_USER=Xihe666
: "${HF_USER:?Set HF_USER to your Hugging Face username before running this script}"

CUDA_VISIBLE_DEVICES=0 /usr/local/bin/torchrun --standalone --nproc_per_node=1 -m lerobot.scripts.lerobot_train \
  --resume=true \
  --config_path=outputs/train/drifting_siemens_0722/checkpoints/010000/pretrained_model/train_config.json \
  --steps=10002 \
  --output_dir=outputs/train/drifting_siemens_0722 \
  --policy.repo_id="${HF_USER}/drifting_siemens_0722"