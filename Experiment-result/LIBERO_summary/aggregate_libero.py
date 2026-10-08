#!/usr/bin/env python3
"""Aggregate LIBERO latency benchmark results across seeds.

For every LIBERO task it scans the per-run timing CSVs, averages the per-step
`backbone_ms` / `action_head_ms` / `total_ms` of each run, then averages those
run-level means across the seeds of each model.

Sources
-------
* ``latency_bench/seed_compare/<task>/`` : multi-seed runs (20k checkpoints)
  -> drif_ov (seed1, seed42, seed1000) and gr00t (whatever seeds are present).
* ``latency_bench/<task>/``              : single-run pi0.5 baseline (seed1000).

Success rates are *not* recomputable from the CSVs; they are read from the
generated ``summary.json`` next to each task.  When that file is empty/missing
the success rate is reported as ``N/A``.

Outputs (written next to this script)
-------------------------------------
* ``libero_<task>_summary.csv``  : one row per model (mean over seeds)
* ``libero_all_summary.csv``     : all tasks concatenated
* ``libero_all_seeds.csv``       : per-seed detail (run-level rows)
* ``LIBERO_summary.md``          : human readable report

Usage
-----
    python Experiment-result/LIBERO_summary/aggregate_libero.py
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent

TASKS = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]

# Canonical display names keyed by a substring found in the file / model column.
MODEL_ALIASES = [
    ("drif_ov", "drif_ov"),
    ("drifting", "drifting"),
    ("gr00t", "gr00t"),
    ("groot", "gr00t"),
    ("pi05", "pi05"),
]

# The gr00t seed1000 runs predate the ``*_20k_seed*`` naming convention: they
# live in ``LIBERO_latency_stats/`` under ``Xihe666_gr00t_n17_libero`` (spatial
# was archived as ``..._1A800_0811``).  Their success rate is recorded in that
# folder's summary.json under the key ``Xihe666/gr00t_n17_libero``.
GR00T_SEED1000_CSV = {
    "libero_spatial": "Xihe666_gr00t_n17_libero_1A800_0811_timings.csv",
    "libero_goal": "Xihe666_gr00t_n17_libero_timings.csv",
    "libero_object": "Xihe666_gr00t_n17_libero_timings.csv",
    "libero_10": "Xihe666_gr00t_n17_libero_timings.csv",
}
GR00T_SEED1000_SUMMARY_KEY = "Xihe666/gr00t_n17_libero"

_SEED_RE = re.compile(r"seed(\d+)")


def canonical_model(name: str) -> str:
    low = name.lower()
    for needle, alias in MODEL_ALIASES:
        if needle in low:
            return alias
    return low


def find_seed(name: str) -> str:
    m = _SEED_RE.search(name)
    # `_seed1000_` / `seed42` / `seed1` -> "1000" / "42" / "1"
    return m.group(1) if m else "unknown"


def read_timing_csv(path: Path) -> dict | None:
    """Return run-level means for backbone / action-head / total ms."""
    backbone, head, total = [], [], []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                backbone.append(float(row["backbone_ms"]))
                head.append(float(row["action_head_ms"]))
                total.append(float(row["total_ms"]))
            except (KeyError, ValueError, TypeError):
                continue
    if not backbone:
        return None
    n = len(backbone)
    return {
        "backbone_ms": sum(backbone) / n,
        "action_head_ms": sum(head) / n,
        "total_ms": sum(total) / n,
        "n_timing_calls": n,
    }


def read_summary_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    # Keys are "<owner>/<policy>"; index by the policy suffix and by the
    # flattened "<owner>_<policy>" form used for the CSV filenames.
    out = {}
    for key, value in data.items():
        out[key] = value
        out.setdefault(key.rsplit("/", 1)[-1], value)
        out.setdefault(key.replace("/", "_"), value)
    return out


def collect_runs(task: str) -> list[dict]:
    """Collect every individual run (one model, one seed) for a task."""
    runs: list[dict] = []

    def add(csv_path: Path, summary: dict, source: str) -> bool:
        stats = read_timing_csv(csv_path)
        if stats is None:
            return False
        stem = csv_path.name[: -len("_timings.csv")]
        model = canonical_model(stem)
        seed = find_seed(stem)
        if seed == "unknown" and model == "pi05":
            # The pi0.5 checkout is the seed1000 20k run even though the
            # filename carries no seed token.
            seed = "1000"
        meta = summary.get(stem, {})
        success = meta.get("success_rate")
        # 0.0 with no timing evidence is a placeholder, keep it as-is when
        # the model truly failed to produce timings, else trust the value.
        runs.append(
            {
                "task": task,
                "model": model,
                "seed": seed,
                "source": source,
                "run": stem,
                **stats,
                "success_rate": success if success is not None else None,
            }
        )
        return True

    seed_dir = EXPERIMENT_ROOT / "latency_bench" / "seed_compare" / task
    if seed_dir.is_dir():
        summary = read_summary_json(seed_dir / "summary.json")
        for csv_path in sorted(seed_dir.glob("*_timings.csv")):
            add(csv_path, summary, "seed_compare")

    base_dir = EXPERIMENT_ROOT / "latency_bench" / task
    if base_dir.is_dir():
        summary = read_summary_json(base_dir / "summary.json")
        for csv_path in sorted(base_dir.glob("*_timings.csv")):
            add(csv_path, summary, "latency_bench")

    # gr00t seed1000 archived under LIBERO_latency_stats/.
    seed1000_csv = GR00T_SEED1000_CSV.get(task)
    legacy_dir = EXPERIMENT_ROOT / "LIBERO_latency_stats" / task
    if seed1000_csv and (legacy_dir / seed1000_csv).exists():
        stats = read_timing_csv(legacy_dir / seed1000_csv)
        if stats is not None:
            summary = read_summary_json(legacy_dir / "summary.json")
            success = summary.get(GR00T_SEED1000_SUMMARY_KEY, {}).get("success_rate")
            stem = seed1000_csv[: -len("_timings.csv")]
            runs.append(
                {
                    "task": task,
                    "model": "gr00t",
                    "seed": "1000",
                    "source": "LIBERO_latency_stats",
                    "run": stem,
                    **stats,
                    "success_rate": success if success is not None else None,
                }
            )

    return runs


def collect_legacy() -> list[dict]:
    """Older single-run results under ``LIBERO_latency_stats/`` (reference only).

    Runs already consumed as gr00t seed1000 are excluded.
    """
    runs: list[dict] = []
    consumed = set(GR00T_SEED1000_CSV.values())
    for task in TASKS:
        directory = EXPERIMENT_ROOT / "LIBERO_latency_stats" / task
        if not directory.is_dir():
            continue
        summary = read_summary_json(directory / "summary.json")
        for csv_path in sorted(directory.glob("*_timings.csv")):
            if csv_path.name in consumed:
                continue
            stats = read_timing_csv(csv_path)
            if stats is None:
                continue
            stem = csv_path.name[: -len("_timings.csv")]
            meta = summary.get(stem, {})
            success = meta.get("success_rate")
            runs.append(
                {
                    "task": task,
                    "run": stem,
                    "model": canonical_model(stem),
                    **stats,
                    "success_rate": success if success is not None else None,
                }
            )
    return runs


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def aggregate(runs: list[dict]) -> list[dict]:
    """Group run-level rows by model and average across seeds."""
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        grouped.setdefault(run["model"], []).append(run)

    # Keep a stable, meaningful model order.
    order = {"pi05": 0, "gr00t": 1, "drif_ov": 2, "drifting": 3}
    rows = []
    for model in sorted(grouped, key=lambda m: (order.get(m, 99), m)):
        seeds = sorted(grouped[model], key=lambda r: _seed_sort_key(r["seed"]))
        backbone = [r["backbone_ms"] for r in seeds]
        head = [r["action_head_ms"] for r in seeds]
        total = [r["total_ms"] for r in seeds]
        success = [r["success_rate"] for r in seeds if r["success_rate"] is not None]
        rows.append(
            {
                "model": model,
                "n_seeds": len(seeds),
                "seeds": "+".join(r["seed"] for r in seeds),
                "backbone_ms": mean(backbone),
                "action_head_ms": mean(head),
                "total_ms": mean(total),
                "success_rate": mean(success) if success else None,
                "backbone_ms_std": std(backbone),
                "action_head_ms_std": std(head),
                "total_ms_std": std(total),
                "success_rate_std": std(success) if len(success) > 1 else None,
                "n_timing_calls": sum(r["n_timing_calls"] for r in seeds),
                "n_success_reported": len(success),
            }
        )
    return rows


def _seed_sort_key(seed: str) -> tuple[int, str]:
    return (0, f"{int(seed):09d}") if seed.isdigit() else (1, seed)


def fmt(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _round_row(row: dict) -> dict:
    return {key: round(value, 3) if isinstance(value, float) else value for key, value in row.items()}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_round_row(row) for row in rows)
    print(f"[aggregate] wrote {path.relative_to(EXPERIMENT_ROOT)}")


TASK_CSV_FIELDS = [
    "model",
    "n_seeds",
    "seeds",
    "backbone_ms",
    "action_head_ms",
    "total_ms",
    "success_rate",
    "backbone_ms_std",
    "action_head_ms_std",
    "total_ms_std",
    "success_rate_std",
    "n_timing_calls",
    "n_success_reported",
]

SEED_CSV_FIELDS = [
    "task",
    "model",
    "seed",
    "source",
    "backbone_ms",
    "action_head_ms",
    "total_ms",
    "success_rate",
    "n_timing_calls",
]

ALL_CSV_FIELDS = ["task", *TASK_CSV_FIELDS]

LEGACY_CSV_FIELDS = [
    "task",
    "model",
    "run",
    "backbone_ms",
    "action_head_ms",
    "total_ms",
    "success_rate",
    "n_timing_calls",
]


def main() -> None:
    all_runs: list[dict] = []
    per_task: dict[str, list[dict]] = {}
    all_rows: list[dict] = []

    for task in TASKS:
        runs = collect_runs(task)
        if not runs:
            print(f"[aggregate] WARNING: no runs found for {task}")
            continue
        rows = aggregate(runs)
        per_task[task] = rows
        all_runs.extend(runs)
        for row in rows:
            all_rows.append({"task": task, **row})
        write_csv(HERE / f"{task}_summary.csv", rows, TASK_CSV_FIELDS)

    write_csv(HERE / "libero_all_summary.csv", all_rows, ALL_CSV_FIELDS)
    write_csv(HERE / "libero_all_seeds.csv", all_runs, SEED_CSV_FIELDS)

    legacy = collect_legacy()
    if legacy:
        write_csv(HERE / "libero_legacy_single_runs.csv", legacy, LEGACY_CSV_FIELDS)

    write_markdown(per_task, all_runs, legacy)


def write_markdown(
    per_task: dict[str, list[dict]],
    all_runs: list[dict],
    legacy: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("# LIBERO 实验结果汇总（多 seed 平均）")
    lines.append("")
    lines.append(
        "本目录汇总 `Experiment-result/` 下所有 LIBERO 评测的推理延迟与成功率，"
        "并对每个任务的每个模型按 seed 求平均。原始目录（`latency_bench/`、"
        "`LIBERO_latency_stats/`）保持不变。"
    )
    lines.append("")
    lines.append("## 数据来源与口径")
    lines.append("")
    lines.append("| 模型 | 数据来源 | seed | 说明 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| pi05 | `latency_bench/<task>/` | seed1000 | 20k 训练，仅一个 seed |")
    lines.append(
        "| gr00t | seed1/seed42: `latency_bench/seed_compare/<task>/`；seed1000: `LIBERO_latency_stats/<task>/` | 1 / 42 / 1000 | 20k 训练；seed1000 为最早测试、无 seed 标注的 `Xihe666_gr00t_n17_libero` |"
    )
    lines.append("| drif_ov | `latency_bench/seed_compare/<task>/` | seed1 / seed42 / seed1000 | 20k 训练 |")
    lines.append("")
    lines.append(
        "- 单次运行的 `backbone_ms` / `action_head_ms` / `total_ms` 为逐 step CSV 的算术平均；"
        "模型数值为各 seed 运行均值的算术平均（等权）。"
    )
    lines.append("- `std` 为样本标准差（ddof=1）；单个 seed 时为 `N/A`。")
    lines.append(
        "- 成功率来自各 `summary.json` 中的 `pc_success`，按 seed 等权平均。"
        "`latency_bench/seed_compare/libero_object/summary.json` 为空，"
        "因此 object 中 seed1/seed42 运行的成功率为 `N/A`；"
        "若只有部分 seed 有成功率，表中以 `(k/n seeds)` 标注。"
    )
    lines.append(
        "- 注意：pi05 的计时 CSV 与同目录 `latency_bench/<task>/summary.json` 在 "
        "spatial / object / libero_10 上不一致（summary 是更早一次运行的残留值）。"
        "本表按约定以 CSV 重算为准。"
    )
    lines.append(
        "- 注意：`libero_spatial` 的 gr00t seed1000 延迟取自 "
        "`LIBERO_latency_stats/libero_spatial/Xihe666_gr00t_n17_libero_1A800_0811_timings.csv`"
        "（该目录没有不带后缀的 `gr00t_n17_libero` CSV），成功率取自同目录 summary.json "
        "的 `gr00t_n17_libero`（80.0，n=211）；两者采样数略有差异（220 vs 211）。"
    )
    lines.append("")
    lines.append("## 各任务结果")
    lines.append("")

    for task in TASKS:
        if task not in per_task:
            continue
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| model | seeds | backbone (ms) | action head (ms) | total (ms) | success (%) |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in per_task[task]:
            total = f"{row['total_ms']:.2f}"
            if row["total_ms_std"] is not None:
                total += f" ± {row['total_ms_std']:.2f}"
            success = fmt(row["success_rate"], 1)
            if row["success_rate"] is not None and row["success_rate_std"] is not None:
                success += f" ± {row['success_rate_std']:.1f}"
            if row["success_rate"] is not None and row["n_success_reported"] < row["n_seeds"]:
                success += f" ({row['n_success_reported']}/{row['n_seeds']} seeds)"
            lines.append(
                f"| {row['model']} | {row['seeds']} | "
                f"{row['backbone_ms']:.2f} | {row['action_head_ms']:.2f} | "
                f"{total} | {success} |"
            )
        lines.append("")
        # Per-seed detail
        task_runs = [r for r in all_runs if r["task"] == task]
        lines.append("<details><summary>per-seed 明细</summary>")
        lines.append("")
        lines.append(
            "| model | seed | source | backbone (ms) | action head (ms) | total (ms) | success (%) | n_calls |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for run in sorted(task_runs, key=lambda r: (r["model"], _seed_sort_key(r["seed"]))):
            lines.append(
                f"| {run['model']} | {run['seed']} | {run['source']} | "
                f"{run['backbone_ms']:.2f} | {run['action_head_ms']:.2f} | "
                f"{run['total_ms']:.2f} | {fmt(run['success_rate'], 1)} | "
                f"{run['n_timing_calls']} |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Gap report
    lines.append("## 数据缺口")
    lines.append("")
    lines.append(
        "需求是「每个模型 3 个 seed 平均」（pi05 只有一个 seed，属预期）。多 seed 模型当前覆盖情况如下："
    )
    lines.append("")
    lines.append("| task | model | 现有 seed | 缺失 |")
    lines.append("| --- | --- | --- | --- |")
    gaps = 0
    for task in TASKS:
        if task not in per_task:
            continue
        for row in per_task[task]:
            if row["model"] == "pi05":
                continue
            have = set(row["seeds"].split("+"))
            missing = [s for s in ("1", "42", "1000") if s not in have]
            if missing:
                gaps += 1
            lines.append(
                f"| {task} | {row['model']} | {row['seeds']} | {'、'.join(missing) if missing else '无'} |"
            )
    lines.append("")
    if gaps:
        lines.append("把缺失的运行结果放入对应目录后重新运行本脚本即可自动纳入平均：")
    else:
        lines.append("三个 seed 已全部覆盖，无需补充。重新运行本脚本即可刷新：")
    lines.append("")
    lines.append("```bash")
    lines.append("python Experiment-result/LIBERO_summary/aggregate_libero.py")
    lines.append("```")
    lines.append("")
    if legacy:
        lines.append("## 附：早期单次运行（参考，非 3-seed 集合）")
        lines.append("")
        lines.append(
            "`LIBERO_latency_stats/` 下其余早于多 seed 实验的单次运行"
            "（如 `drif_ov_libero0809`、`drifting_libero_full`）不属于上面的"
            " 20k 多 seed 集合，仅作参考，不参与平均。"
            "gr00t 的 `gr00t_n17_libero` 已作为 seed1000 纳入上面的平均。"
        )
        lines.append("")
        lines.append("| task | run | backbone (ms) | action head (ms) | total (ms) | success (%) | n_calls |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for run in sorted(legacy, key=lambda r: (r["task"], r["run"])):
            lines.append(
                f"| {run['task']} | {run['run']} | {run['backbone_ms']:.2f} | "
                f"{run['action_head_ms']:.2f} | {run['total_ms']:.2f} | "
                f"{fmt(run['success_rate'], 1)} | {run['n_timing_calls']} |"
            )
        lines.append("")

    lines.append("## 输出文件")
    lines.append("")
    lines.append("| 文件 | 内容 |")
    lines.append("| --- | --- |")
    lines.append(
        "| `libero_<task>_summary.csv` | 每个 task 一行一个模型的 seed 平均（列名沿用原始 `*_timings.csv`） |"
    )
    lines.append("| `libero_all_summary.csv` | 四个 task 合并 |")
    lines.append("| `libero_all_seeds.csv` | 每个 seed 的逐次运行明细 |")
    lines.append("| `libero_legacy_single_runs.csv` | `LIBERO_latency_stats/` 早期单次运行（参考） |")
    lines.append("| `aggregate_libero.py` | 生成上述文件的脚本 |")
    lines.append("")

    path = HERE / "LIBERO_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[aggregate] wrote {path.relative_to(EXPERIMENT_ROOT)}")


if __name__ == "__main__":
    main()
