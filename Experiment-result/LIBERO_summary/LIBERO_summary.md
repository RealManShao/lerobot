# LIBERO 实验结果汇总（多 seed 平均）

本目录汇总 `Experiment-result/` 下所有 LIBERO 评测的推理延迟与成功率，并对每个任务的每个模型按 seed 求平均。原始目录（`latency_bench/`、`LIBERO_latency_stats/`）保持不变。

## 数据来源与口径

| 模型 | 数据来源 | seed | 说明 |
| --- | --- | --- | --- |
| pi05 | `latency_bench/<task>/` | seed1000 | 20k 训练，仅一个 seed |
| gr00t | seed1/seed42: `latency_bench/seed_compare/<task>/`；seed1000: `LIBERO_latency_stats/<task>/` | 1 / 42 / 1000 | 20k 训练；seed1000 为最早测试、无 seed 标注的 `Xihe666_gr00t_n17_libero` |
| drif_ov | `latency_bench/seed_compare/<task>/` | seed1 / seed42 / seed1000 | 20k 训练 |

- 单次运行的 `backbone_ms` / `action_head_ms` / `total_ms` 为逐 step CSV 的算术平均；模型数值为各 seed 运行均值的算术平均（等权）。
- `std` 为样本标准差（ddof=1）；单个 seed 时为 `N/A`。
- 成功率来自各 `summary.json` 中的 `pc_success`，按 seed 等权平均。`latency_bench/seed_compare/libero_object/summary.json` 为空，因此 object 中 seed1/seed42 运行的成功率为 `N/A`；若只有部分 seed 有成功率，表中以 `(k/n seeds)` 标注。
- 注意：pi05 的计时 CSV 与同目录 `latency_bench/<task>/summary.json` 在 spatial / object / libero_10 上不一致（summary 是更早一次运行的残留值）。本表按约定以 CSV 重算为准。
- 注意：`libero_spatial` 的 gr00t seed1000 延迟取自 `LIBERO_latency_stats/libero_spatial/Xihe666_gr00t_n17_libero_1A800_0811_timings.csv`（该目录没有不带后缀的 `gr00t_n17_libero` CSV），成功率取自同目录 summary.json 的 `gr00t_n17_libero`（80.0，n=211）；两者采样数略有差异（220 vs 211）。

## 各任务结果

### libero_spatial

| model | seeds | backbone (ms) | action head (ms) | total (ms) | success (%) |
| --- | --- | --- | --- | --- | --- |
| pi05 | 1000 | 38.87 | 72.96 | 111.83 | 0.0 |
| gr00t | 1+42+1000 | 24.99 | 45.36 | 70.35 ± 0.19 | 82.3 ± 6.8 |
| drif_ov | 1+42+1000 | 24.42 | 4.96 | 29.38 ± 0.05 | 64.0 ± 4.0 |

<details><summary>per-seed 明细</summary>

| model | seed | source | backbone (ms) | action head (ms) | total (ms) | success (%) | n_calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| drif_ov | 1 | seed_compare | 24.54 | 4.90 | 29.43 | 60.0 | 470 |
| drif_ov | 42 | seed_compare | 24.39 | 4.94 | 29.33 | 64.0 | 457 |
| drif_ov | 1000 | seed_compare | 24.33 | 5.06 | 29.38 | 68.0 | 444 |
| gr00t | 1 | seed_compare | 24.67 | 45.49 | 70.15 | 77.0 | 401 |
| gr00t | 42 | seed_compare | 25.12 | 45.42 | 70.54 | 90.0 | 361 |
| gr00t | 1000 | LIBERO_latency_stats | 25.19 | 45.17 | 70.36 | 80.0 | 220 |
| pi05 | 1000 | latency_bench | 38.87 | 72.96 | 111.83 | 0.0 | 50 |

</details>

### libero_goal

| model | seeds | backbone (ms) | action head (ms) | total (ms) | success (%) |
| --- | --- | --- | --- | --- | --- |
| pi05 | 1000 | 77.20 | 243.47 | 320.67 | 0.0 |
| gr00t | 1+42+1000 | 25.03 | 45.43 | 70.46 ± 0.50 | 82.3 ± 4.0 |
| drif_ov | 1+42+1000 | 26.47 | 5.07 | 31.54 ± 3.97 | 52.0 ± 1.0 |

<details><summary>per-seed 明细</summary>

| model | seed | source | backbone (ms) | action head (ms) | total (ms) | success (%) | n_calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| drif_ov | 1 | seed_compare | 24.15 | 4.79 | 28.95 | 52.0 | 542 |
| drif_ov | 42 | seed_compare | 24.53 | 5.04 | 29.57 | 51.0 | 526 |
| drif_ov | 1000 | seed_compare | 30.72 | 5.39 | 36.11 | 53.0 | 523 |
| gr00t | 1 | seed_compare | 25.06 | 45.65 | 70.71 | 80.0 | 419 |
| gr00t | 42 | seed_compare | 24.56 | 45.32 | 69.89 | 87.0 | 382 |
| gr00t | 1000 | LIBERO_latency_stats | 25.47 | 45.33 | 70.80 | 80.0 | 210 |
| pi05 | 1000 | latency_bench | 77.20 | 243.47 | 320.67 | 0.0 | 300 |

</details>

### libero_object

| model | seeds | backbone (ms) | action head (ms) | total (ms) | success (%) |
| --- | --- | --- | --- | --- | --- |
| pi05 | 1000 | 33.29 | 65.49 | 98.78 | 0.0 |
| gr00t | 1+42+1000 | 26.21 | 45.80 | 72.01 ± 1.51 | 88.0 (1/3 seeds) |
| drif_ov | 1+42+1000 | 24.30 | 4.94 | 29.24 ± 0.16 | N/A |

<details><summary>per-seed 明细</summary>

| model | seed | source | backbone (ms) | action head (ms) | total (ms) | success (%) | n_calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| drif_ov | 1 | seed_compare | 24.47 | 4.96 | 29.42 | N/A | 506 |
| drif_ov | 42 | seed_compare | 24.23 | 4.97 | 29.21 | N/A | 539 |
| drif_ov | 1000 | seed_compare | 24.21 | 4.89 | 29.11 | N/A | 526 |
| gr00t | 1 | seed_compare | 25.86 | 45.70 | 71.56 | N/A | 409 |
| gr00t | 42 | seed_compare | 24.99 | 45.78 | 70.77 | N/A | 418 |
| gr00t | 1000 | LIBERO_latency_stats | 27.77 | 45.92 | 73.69 | 88.0 | 220 |
| pi05 | 1000 | latency_bench | 33.29 | 65.49 | 98.78 | 0.0 | 269 |

</details>

### libero_10

| model | seeds | backbone (ms) | action head (ms) | total (ms) | success (%) |
| --- | --- | --- | --- | --- | --- |
| pi05 | 1000 | 34.21 | 68.10 | 102.31 | 0.0 |
| gr00t | 1+1000 | 24.14 | 45.18 | 69.33 ± 0.36 | 74.5 ± 10.6 |
| drif_ov | 1+42+1000 | 25.92 | 5.04 | 30.95 ± 3.72 | 26.0 ± 2.6 |

<details><summary>per-seed 明细</summary>

| model | seed | source | backbone (ms) | action head (ms) | total (ms) | success (%) | n_calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| drif_ov | 1 | seed_compare | 23.99 | 4.91 | 28.90 | 28.0 | 1126 |
| drif_ov | 42 | seed_compare | 23.85 | 4.86 | 28.71 | 27.0 | 1110 |
| drif_ov | 1000 | seed_compare | 29.90 | 5.34 | 35.25 | 23.0 | 1163 |
| gr00t | 1 | seed_compare | 23.84 | 45.23 | 69.07 | 67.0 | 895 |
| gr00t | 1000 | LIBERO_latency_stats | 24.45 | 45.13 | 69.58 | 82.0 | 419 |
| pi05 | 1000 | latency_bench | 34.21 | 68.10 | 102.31 | 0.0 | 536 |

</details>

## 数据缺口

需求是「每个模型 3 个 seed 平均」（pi05 只有一个 seed，属预期）。多 seed 模型当前覆盖情况如下：

| task | model | 现有 seed | 缺失 |
| --- | --- | --- | --- |
| libero_spatial | gr00t | 1+42+1000 | 无 |
| libero_spatial | drif_ov | 1+42+1000 | 无 |
| libero_goal | gr00t | 1+42+1000 | 无 |
| libero_goal | drif_ov | 1+42+1000 | 无 |
| libero_object | gr00t | 1+42+1000 | 无 |
| libero_object | drif_ov | 1+42+1000 | 无 |
| libero_10 | gr00t | 1+1000 | 42 |
| libero_10 | drif_ov | 1+42+1000 | 无 |

把缺失的运行结果放入对应目录后重新运行本脚本即可自动纳入平均：

```bash
python Experiment-result/LIBERO_summary/aggregate_libero.py
```

## 附：早期单次运行（参考，非 3-seed 集合）

`LIBERO_latency_stats/` 下其余早于多 seed 实验的单次运行（如 `drif_ov_libero0809`、`drifting_libero_full`）不属于上面的 20k 多 seed 集合，仅作参考，不参与平均。gr00t 的 `gr00t_n17_libero` 已作为 seed1000 纳入上面的平均。

| task | run | backbone (ms) | action head (ms) | total (ms) | success (%) | n_calls |
| --- | --- | --- | --- | --- | --- | --- |
| libero_10 | Xihe666_drif_ov_libero0809 | 24.25 | 5.03 | 29.28 | 46.0 | 498 |
| libero_goal | Xihe666_drif_ov_libero0809 | 24.38 | 4.84 | 29.22 | 80.0 | 221 |
| libero_object | Xihe666_drif_ov_libero0809 | 24.56 | 5.00 | 29.56 | 74.0 | 241 |
| libero_spatial | Xihe666_drif_ov_libero0809 | 24.35 | 4.90 | 29.25 | 72.0 | 220 |
| libero_spatial | Xihe666_drifting_libero_full | 24.67 | 4.81 | 29.48 | N/A | 205 |

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `libero_<task>_summary.csv` | 每个 task 一行一个模型的 seed 平均（列名沿用原始 `*_timings.csv`） |
| `libero_all_summary.csv` | 四个 task 合并 |
| `libero_all_seeds.csv` | 每个 seed 的逐次运行明细 |
| `libero_legacy_single_runs.csv` | `LIBERO_latency_stats/` 早期单次运行（参考） |
| `aggregate_libero.py` | 生成上述文件的脚本 |
