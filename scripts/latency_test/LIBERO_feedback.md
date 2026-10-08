# LIBERO Long / LIBERO 90：固定 H=40，比较不同 K 的反馈收益

本实验使用现有 `Xihe666/drif_ov_libero_20k_seed42_0821` checkpoint，不训练模型。
每次预测完整 40 步动作，立即用本次观测对应的状态解码整个动作块，只执行前
K 步，然后用最新观测重新规划。默认 K 为 1、5、10、20、40。

`chunk_size` 和 `n_action_steps` 均保持 40；后者也会限制模型返回的预测长度，
因此不能用它设置 K。脚本在外部控制执行长度，并验证每次输出确实为 `(40, 7)`。
本轮使用同步规划、空动作前缀，不加入 overlap、动作混合或异步推理。

## 运行

在仓库根目录执行。环境为现有 `conda/lerobot`，需要可用的 NVIDIA GPU、
LIBERO 资源及 checkpoint。无显示器时脚本默认使用 EGL。

```bash
conda activate lerobot

# 检查实验配置，不加载模型，不创建实验数据。
python scripts/latency_test/benchmark_libero_feedback.py --dry-run

# LIBERO 90：全部 90 个任务，每个任务、每个 K 各 5 回合。
# 共 2250 回合，每个 K 为 450 回合；标准上限自动设为 400 步。
python scripts/latency_test/benchmark_libero_feedback.py \
  --suite libero_90 --n-episodes 5 --local-files-only \
  --output-dir outputs/libero90_feedback_5ep

# LIBERO 90 中断后续跑：保持原始参数并增加 --resume。
python scripts/latency_test/benchmark_libero_feedback.py \
  --suite libero_90 --n-episodes 5 --local-files-only \
  --output-dir outputs/libero90_feedback_5ep --resume

# 冒烟验证：1 个任务、每种 K 一个回合、最多 45 步。
# 这能覆盖 K=40 的第一次块边界，但不能用于正式成功率结论。
python scripts/latency_test/benchmark_libero_feedback.py \
  --task-ids 0 --n-episodes 1 --max-steps 45 \
  --output-dir outputs/libero_feedback_smoke

# 正式实验：全部 10 个 Long 任务，每个 K 每个任务 10 回合，共 500 回合。
python scripts/latency_test/benchmark_libero_feedback.py \
  --n-episodes 10 --output-dir outputs/libero_feedback

# 中断后续跑：保留所有原始参数，只增加 --resume。
python scripts/latency_test/benchmark_libero_feedback.py \
  --n-episodes 10 --output-dir outputs/libero_feedback --resume

# 从已完成回合重新汇总、绘图；无需 GPU、LIBERO 或模型加载。
python scripts/latency_test/libero_feedback_report.py \
  --output-dir outputs/libero_feedback
```

可以使用 `conda run --no-capture-output -n lerobot python ...` 替代激活环境。
模型已在本地时加 `--checkpoint /absolute/path/to/pretrained_model`，目录需要包含
配置、权重及 processor 文件。HF 下载缓存齐全时可以使用 `--local-files-only`，
脚本同时设置 `HF_HUB_OFFLINE=1`，禁止包括 backbone/tokenizer 在内的依赖联网。
远端模型首次下载后会记录精确 commit，续跑固定这个版本。

缺少首次运行的 LIBERO 配置时，脚本自动创建实验目录内的 `libero_config/config.yaml`，
避免在无人值守实验中出现交互提示；已有 `LIBERO_CONFIG_PATH` 或 `~/.libero/config.yaml`
会被复用。仿真 assets 默认使用安装目录或 `~/.cache/libero/assets`，也可通过
`--libero-assets /path/to/assets` 指定。缺少默认 assets 时，脚本调用当前 HF API 下载，
避开安装版 LIBERO 内置下载函数中的过时参数。

本 checkpoint 引用 `nvidia/Cosmos-Reason2-2B`。联网环境可以提前准备以下资源，
无需下载 LIBERO 训练数据集：

```bash
hf download Xihe666/drif_ov_libero_20k_seed42_0821
hf download nvidia/Cosmos-Reason2-2B
hf download lerobot/libero-assets --repo-type dataset \
  --local-dir "$HOME/.cache/libero/assets"
```

默认使用 `libero_10`（LIBERO Long）的全部 10 个任务、520 步上限；
`--suite libero_90` 自动切换到全部 90 个任务和 400 步上限。
两者都使用 360×360 双相机图像，与现有 `lerobot-eval` 的图像设置相同。
`--n-episodes` 始终表示每个任务、每个 K 的回合数。
使用 `--task-ids` 或缩短 `--max-steps` 时，图中会标注任务子集或 SMOKE。

## 公平比较

- 同一任务、同一回合在所有 K 下使用相同 LIBERO 初始状态索引和随机种子。
  仅重设 seed 不会重设 LIBERO 初始状态，脚本会显式重设 `init_state_id`。
- 每个评估回合新建环境，防止安装版 LIBERO 多次 hard reset 时累积
  Open/Close/TurnOn 初始化采样器，改变 fixture 位置。开始执行前计算原始双相机图像
  和机器人状态的 SHA256；同一任务、回合在各 K 下必须完全一致，否则终止实验。
  指纹写入回合记录，断点续跑也会对照已保存的指纹。schema 2 禁止混用旧 reset 协议的数据。
- 每个任务先做三次不计时的完整预测；之后重新 reset 环境、policy 和 processors，
  丢弃 warmup 环境，再创建评估环境并重设随机种子。不同 K 的执行顺序按固定种子打乱，
  减少运行顺序造成的偏差。
- 每次只运行一个环境；不同 K 共用同一个已加载模型。
- 不计模型加载、下载、reset、warmup、报告和磁盘写入时间。不录制视频。
  逐动作记录先留在内存，每回合结束后原子保存，避免磁盘写入污染时序。
- 修正现有 LIBERO 封装在终止时自动 reset 的计时问题，仅在脚本内的子类中去除
  terminal autoreset。库的正常评估行为不变。
- 每种 K 的推理次数不同，随机模型在整个回合中不会消耗完全相同的随机数序列。
  配对的是初始条件和初始随机种子。

## 控制时钟与指标定义

控制频率从运行中的 LIBERO `control_freq` 读取，通常为 20 Hz（周期 T=50 ms）。
不使用视频帧率或配置中的 `fps` 代替控制频率。可以显式给 `--control-hz 20`，
但它必须与环境实际频率一致。

在初始观测进入测量循环时启动计时。首个动作的 deadline 为开始时间加 T，之后
每个动作的 deadline 为上一个动作**实际发送时间**加 T。新块在上一动作的
`env.step` 返回后立即开始规划；动作准备好且到达 deadline 后发送。
超时就立即发送，并从实际发送时间安排下一周期，不突发补发动作。
默认允许 `--deadline-tolerance-ms 1` 的调度容差，原始超时毫秒数仍完整保留。

| 指标 | 定义及输出 |
|---|---|
| 成功率 | 成功回合数 / 已完成评估回合数；同时给出任务分项、样本量、描述性 Wilson 95% 区间。中断回合不算失败。|
| 完成实际秒数 | 从初始观测进入测量循环，到成功动作的 `env.step` 返回，包含推理、仿真计算和控制节拍等待。成功回合单独汇总；失败回合仅进入全部回合耗时统计。|
| 仿真秒数 | 执行动作数 / 环境控制频率；与实际墙钟耗时分别报告。|
| 每动作观测年龄 | 动作发送时间减去生成该动作块所用观测的交付时间（ms）。同时记录仿真年龄，即该块已执行动作数 0…K−1。|
| 端到端延迟 | 源观测交付到该块第一个动作发送（ms），包含预处理、GPU 推理、后处理、CPU 拷贝及等待 deadline 的时间。另报 `pipeline_ms`，隔离完整计算流水线耗时。|
| 块边界动作抖动 | 新块首动作与上一已执行动作之间，前六个机械臂控制分量的差分 L2 范数；同时报告平移三维、旋转三维、二阶动作差分和夹爪符号切换比例。与块内差分对比。|
| 块边界时间抖动 | 块边界相邻实际发送间隔与 T 的偏差；原始记录保留符号，图汇总绝对偏差。|
| 错过 deadline 比例 | 实际发送时间 > 本动作 deadline + 容差的动作数 / 全部实际发送动作数。包含 OS sleep overshoot；另报动作准备阶段的超时比例。|

观测时间戳是模拟器将图像和状态交付给控制循环的时间，不是硬件相机曝光时间；
渲染成本进入 `env_step_ms`、整体墙钟时间和后续 deadline 预算。
CUDA 在流水线内同步，最终动作已经完成 CPU 转移，不把异步 kernel 提交耗时当推理耗时。
`pipeline_ms` 对每次规划加权，观测年龄和 deadline 比例对每个实际发送动作加权。
各项连续指标包含 count、mean、p50、p95、p99、max。

机械臂六维控制分量的范数是**动作命令的变化量**，包含正常的有意运动，不能解释成
物理空间中的速度、加速度或 jerk；平移与旋转也不是相同物理单位。
episode 第一个动作没有前驱，不计入边界；K=1 没有块内样本，对应值为 null。
夹爪离散开合独立统计，避免它主导机械臂差分范数。

LIBERO 是步进仿真：规划和 sleep 时物理世界暂停，因此本实验测的是同步部署下
反馈粒度、成功率及计算/控制时间成本，不能据此证明真实机器人在推理停顿期间的
动态鲁棒性。后者需要另做物理世界持续推进、定义 hold/drop 策略的异步控制实验。
deadline 比例的分母是实际发送动作，不是虚构的理想时钟槽位；一个很长的停顿
算一个错过 deadline 的动作，同时由 lateness、墙钟时间和观测年龄反映其严重程度。

## 输出及判读

```text
outputs/libero_feedback/
  metadata.json             实验配置、计时协议（续跑参数必须一致）
  runtime.json              checkpoint commit、GPU、PyTorch、代码 commit
  libero_resources.json     实际使用的 LIBERO 配置和 assets 路径
  episodes/*.json           每回合原始逐动作、逐规划记录（原子写入）
  summary.json / summary.csv
  per_task_summary.csv      每个 K × 每个任务
  episodes.csv / actions.csv / plans.csv
  comparison.png / comparison.pdf
  boundary_diagnostics.png / boundary_diagnostics.pdf
```

报告每个任务完成后更新，退出时也会汇总已经完成的回合。不完整实验显示
`PARTIAL DATA`；无样本显示空值，不能把空值视为零成功率或零延迟。
只跑完部分任务时，各 K 的任务构成可能不同，应补齐实验后再下结论。

重点看较小 K 是否降低观测年龄并提升成功率，以及为此付出的真实耗时、超时和
边界变化代价。成功回合平均耗时会受到成功样本变化影响，因此报告同时提供
`paired_vs_k40`：只在 K 与 K=40 **都成功的同一初始条件**上计算耗时差和比值，
并记录仅 K 成功、仅 K=40 成功的配对数量。若两边都成功的样本为零，耗时比较保留空值。
Wilson 区间是回合级描述性区间，没有建模同一任务内的相关性。

## 验证

测试使用可控时钟和虚拟环境验证实验调度与统计；这些测试数据不充当模型成绩。

```bash
conda run -n lerobot python -m pytest \
  tests/scripts/test_libero_feedback.py --confcutdir=tests/scripts -q
```
