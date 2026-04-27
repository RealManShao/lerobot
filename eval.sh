lerobot-eval \
--output_dir=./eval_logs/ \
--env.type=libero \
--env.task=libero_spatial,libero_object,libero_goal,libero_10 \
--eval.batch_size=1 \
--eval.n_episodes=10 \
--policy.path=lerobot/pi05_libero_finetuned \
--policy.n_action_steps=10 \
--env.max_parallel_tasks=1


lerobot-imgtransform-viz \
  --repo-id=HuggingFaceVLA/libero \
  --output-dir=./transform_examples \
  --n-examples=5


lerobot-train \
  --dataset.repo_id=namiki26/so101-test-2 \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/my_policy