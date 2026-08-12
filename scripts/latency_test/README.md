# Inference Latency Benchmark

Measures inference latency for Drifting-family policies on LIBERO tasks and
produces two styles of plot:

| Script | What it measures | Plot |
|--------|-----------------|------|
| `benchmark_offline_latency.py` | Offline warm-cache stage-by-stage GPU timing (P50/P99/Max/Mean per stage) | `plot_stage_breakdown.py` |
| `benchmark_libero_latency.py` | Live LIBERO eval — VLM backbone vs action-head split per inference call | `plot_libero_vlm_vs_head.py` |

---

## Quick start

### Offline stage-by-stage benchmark

```bash
# Drifting (Tron2 checkpoint — update --checkpoint / --dataset as needed)
CUDA_VISIBLE_DEVICES=0 python scripts/latency_test/benchmark_offline_latency.py \
    --policy drifting

# Drif-OV (Libero-10 checkpoint)
CUDA_VISIBLE_DEVICES=0 python scripts/latency_test/benchmark_offline_latency.py \
    --policy drif_ov \
    --checkpoint outputs/train/libero/drif_ov/try1/checkpoints/latest/pretrained_model \
    --dataset data/libero-10

# GR00T-N1.7 (downloads from HF Hub on first run)
CUDA_VISIBLE_DEVICES=0 python scripts/latency_test/benchmark_offline_latency.py \
    --policy groot_n17
```

Then fill in the `DATA` dict in `plot_stage_breakdown.py` and regenerate:

```bash
python scripts/latency_test/plot_stage_breakdown.py
# → docs/source/assets/inference_latency_stage_breakdown.png
```

### Live LIBERO eval benchmark (VLM vs head)

```bash
conda run -n lerobot python scripts/latency_test/benchmark_libero_latency.py \
    --policies Xihe666/drifting_libero_full \
               nvidia/gr00t17-lerobot-libero_10-640 \
               Xihe666/drif_ov_libero0809 \
    --task libero_10 --n-episodes 1
# → outputs/latency_bench/libero_10/summary.json
# → outputs/latency_bench/libero_10/libero_10_vlm_vs_head.png
```

Then update `plot_libero_vlm_vs_head.py` with the values from `summary.json` and
regenerate the static plot:

```bash
python scripts/latency_test/plot_libero_vlm_vs_head.py
# → outputs/eval/libero_10_vlm_vs_head.png
```

---

## How the live benchmark works

1. `benchmark_libero_latency.py` calls `lerobot-eval` as a subprocess with
   `LEROBOT_PROFILE_INFERENCE_TIMINGS` set to a per-policy CSV path.
2. Timing hooks inside `DriftingN17.get_action`, `DrifOvN17.get_action`, and
   `GR00TN17.get_action` detect this variable and write one row per step:
   ```
   model,backbone_ms,action_head_ms,total_ms
   drifting,27.043,6.218,33.261
   ```
3. After each run the script reads the CSV, computes means, and parses
   `pc_success` from the eval log.

### Timing scope

| Component | What is timed |
|-----------|--------------|
| VLM / Backbone | `self.backbone(backbone_inputs)` with CUDA sync before/after |
| Action Head | `self.action_head.get_action(...)` with CUDA sync before/after |
| Total | backbone + action head (excludes preprocessing / postprocessing) |

---

## CLI reference

### `benchmark_offline_latency.py`

```
--policy    {drifting,drif_ov,groot_n17}   policy preset (default: drifting)
--checkpoint  PATH                          override checkpoint path or HF repo-id
--dataset     PATH                          override local dataset root
--reps        N                             repetitions per frame (default: 10)
```

### `benchmark_libero_latency.py`

```
--policies  HF_REPO [...]    (required) one or more HF repo paths
--task      libero_10 | libero_spatial | libero_goal | libero_object  (default: libero_10)
--n-episodes N               episodes per sub-task (default: 1)
--output-dir DIR             root for logs, CSVs, and plots (default: outputs/latency_bench)
--rename-map JSON            observation key rename map (auto-applied for gr00t17)
--pytorch-alloc-conf VALUE   PYTORCH_CUDA_ALLOC_CONF (default: expandable_segments:True)
```

---

## Outputs

### Offline benchmark

Prints to stdout only — paste results into `plot_stage_breakdown.py`.

### Live LIBERO benchmark

```
outputs/latency_bench/<task>/
├── <policy>_timings.csv          per-step backbone / action-head split
├── <policy>_eval.log             full lerobot-eval stdout/stderr
├── summary.json                  aggregated means + success rates
└── <task>_vlm_vs_head.png        stacked bar chart (auto-generated)
```

### Example `summary.json`

```json
{
  "Xihe666/drifting_libero_full": {
    "backbone_ms": 27.0,
    "action_head_ms": 6.2,
    "total_ms": 33.2,
    "success_rate": 86.0,
    "n_timing_calls": 100,
    "eval_returncode": 0
  },
  "Xihe666/drif_ov_libero0809": {
    "backbone_ms": 0.0,
    "action_head_ms": 0.0,
    "total_ms": 0.0,
    "success_rate": 49.0,
    "n_timing_calls": 0,
    "eval_returncode": 0
  }
}
```

---

## Adding a new policy

1. Add a timing hook in the model's `get_action` method following the pattern in
   `DrifOvN17.get_action` (`src/lerobot/policies/drif_ov/modeling_drif_ov.py`).
2. Add a preset entry to `_POLICY_DEFAULTS` in `benchmark_offline_latency.py`.
3. If the policy needs a camera key rename for LIBERO, add it to
   `_DEFAULT_RENAME_MAP` in `benchmark_libero_latency.py`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `timing CSV not found` | Timing hook not triggered | Add hook as described above |
| `eval exited with code 1` | OOM or LIBERO env error | Check `<policy>_eval.log`; try `--pytorch-alloc-conf expandable_segments:True` |
| All timings 0.0 ms | `LEROBOT_PROFILE_INFERENCE_TIMINGS` not propagated | Run via `benchmark_libero_latency.py`, not `lerobot-eval` directly |
| GR00T `Repo id must be in form...` | Stale HF cache | Clear `~/.cache/huggingface` and retry |
