# Remote reproduction

Use the checkout at `lerobot-gpu:/root/autodl-tmp/lerobot`. Logs can be redirected into the run directory. The aligned observation cache and frozen visual tower from the preceding experiment are reused. Use a separate result directory for a new run.

```bash
cd /root/autodl-tmp/lerobot
run=outputs/hybrid_v1
cache=outputs/chunk_aligned/cache
suite=/root/autodl-pub/datasets/libero/libero_10
groot=/root/autodl-pub/checkpoints/groot/n1.7-3b
assets=/root/autodl-tmp/libero-runtime/LIBERO-master/libero/libero/assets
uvbin=/root/autodl-tmp/lerobot-env/bin/uv
runpy() { "$uvbin" run --no-sync python "$@"; }
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=/root/autodl-tmp/libero-runtime/egl/usr/lib/x86_64-linux-gnu
export __EGL_VENDOR_LIBRARY_FILENAMES=/root/autodl-tmp/libero-runtime/egl/vendor/10_nvidia.json
export PYTHONPATH=scripts/latency_test:scripts/operation_geometry
```

Train the actuator ablations and the two frozen translation experts for each seed.

```bash
for seed in 1 42; do
  dir="$run/seed$seed"
  for variant in dual_prefix separate_prefix; do
    runpy scripts/one_step/train_demonstration_chunk.py --cache "$cache" --out "$dir" --seed "$seed" --variant "$variant"
  done
  for variant in grip_binary grip_event; do
    runpy scripts/one_step/train_gripper_control.py --root "$dir" --cache "$cache" --seed "$seed" --mode "$variant"
  done
  runpy scripts/one_step/train_demonstration_chunk.py --cache "$cache" --out "$run/expert$seed" --seed "$seed" --variant separate_prefix --reserve-demos 8
  runpy scripts/one_step/train_hybrid_control.py --expert "$run/expert$seed" --out "$dir" --cache "$cache" --suite "$suite" --seed "$seed"
  runpy scripts/one_step/train_history_control.py --root "$dir" --cache "$cache" --suite "$suite" --seed "$seed"
done
```

A rollout uses official initial state IDs. `--episodes` specifies the number per task. All evaluations start a fresh environment per episode. To transplant a gripper, evaluate one demonstration-chunk arm with `--gripper-variant` set to the other model from the same training seed.

```bash
runpy scripts/one_step/eval_libero.py --root "$run/seed1" --variant learned --seed 1 \
  --first-state 6 --episodes 2 --out "$run/seed1/learned-s6" \
  --suite "$suite" --checkpoint "$groot" --libero-assets "$assets"
```

Calibrate consequence-based arbitration after both seeds finish. The first rollout creates the noninteractive LIBERO configuration used by the independent, headless simulator probes. For parallel probing, give each worker a disjoint list of task indices.

```bash
export LIBERO_CONFIG_PATH="$PWD/$run/seed1/learned-s6/libero_config"
runpy scripts/one_step/control_proposals.py --cache "$cache" --root "$run" --out "$run/proposals"
runpy scripts/one_step/control_interventions.py --suite "$suite" --assets "$assets" \
  --proposals "$run/proposals" --out "$run/interventions" --task-ids {0..9}
runpy scripts/one_step/control_interventions.py --suite "$suite" --assets "$assets" \
  --proposals "$run/proposals" --out "$run/oracle" --task-ids {0..9} --oracle-waypoint
runpy scripts/one_step/control_proposals.py --cache "$cache" --root "$run" \
  --history-suite "$suite" --out "$run/history-proposals"
runpy scripts/one_step/control_interventions.py --suite "$suite" --assets "$assets" \
  --proposals "$run/history-proposals" --out "$run/history-interventions" --task-ids {0..9}
for seed in 1 42; do
  runpy scripts/one_step/train_outcome_router.py --cache "$cache" --interventions "$run/interventions" \
    --root "$run/seed$seed" --seed "$seed"
done
```

The motion-alignment diagnostic uses the same frozen history experts. Fit it with the following command for each seed, then evaluate `history_aligned` and `history_matched` on initial states 8–9. The matched constant is estimated on calibration demonstrations.

```bash
runpy scripts/one_step/calibrate_motion_gate.py --cache "$cache" --suite "$suite" --root "$run/seed1" --seed 1
```

Run the variants and initial-state ranges in the experiment report before aggregating. Use `--variant constant --fixed-weight 0.5` to reproduce the explicit 50–50 comparator. The archive preserves the evaluation files used for aggregation.

```bash
runpy scripts/one_step/summarize_hybrid_control.py --root "$run" --out "$run/summary.json"
```
