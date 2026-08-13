# Drifting Inference Latency Analysis - 2026-07-24

## 结论摘要

在 NVIDIA A100-SXM4-80GB 上，Drifting 从三路本地视频帧到一个 40-step action chunk 的 warm-cache 端到端延迟为：

- P50：`81.033 ms`
- P99：`116.262 ms`
- 最大值：`123.771 ms`
- 平均值：`83.486 ms`

模型部分的主要瓶颈是 Cosmos/Qwen VLM backbone，而不是一步生成的 Drifting action head。VLM 的 P50 为 `42.549 ms`，Drifting head 的 P50 为 `13.608 ms`。

![Drifting full-pipeline inference latency](docs/source/assets/drifting/0724_inference_latency.png)

图中每根柱子分别表示 P50、P99、最大值和平均值。柱内颜色是各组件独立计算的同一统计量，黑色菱形是实际测得的端到端延迟。由于各组件的 P99 或最大值不一定发生在同一次采样中，堆叠柱总高度不应被当作端到端 P99 或最大值；应以黑色菱形和下表的端到端数据为准。

## 测试环境

| 项目 | 配置 |
| --- | --- |
| 日期 | 2026-07-24 |
| GPU | NVIDIA A100-SXM4-80GB |
| PyTorch | `2.11.0+cu128` |
| Checkpoint | `outputs/train/drifting_siemens_0722/checkpoints/010002/pretrained_model` |
| Dataset | `data/siemens-v3-disturb` |
| 视频 | 三路 `640x480`、30 FPS、AV1 |
| 视频后端 | TorchCodec `0.11.1+cpu` |
| Action chunk | 40 steps |
| 精度 | VLM/Drifting inference 使用 BF16 autocast |
| 输入帧 | 全局索引 `100`、`100100`、`200100` |
| Episodes | `0`、`98`、`195` |
| 样本数 | 3 个固定帧各重复 10 次，共 30 次 |
| 缓存状态 | 模型、CUDA kernels 和三个 episode decoder 均预热 |

每个 GPU 阶段计时前后都调用 `torch.cuda.synchronize()`。模型加载、Hub 下载和首次 kernel 初始化不计入推理延迟。

## 分阶段结果

| 阶段 | P50 (ms) | P99 (ms) | 最大 (ms) | 平均 (ms) |
| --- | ---: | ---: | ---: | ---: |
| 三相机视频解码 | 6.977 | 8.001 | 8.025 | 7.080 |
| Observation 准备与传入 GPU | 3.827 | 4.301 | 4.325 | 3.851 |
| Policy preprocessor | 13.016 | 17.004 | 17.956 | 13.319 |
| 模型输入整理 | 0.285 | 0.382 | 0.389 | 0.290 |
| VLM backbone | 42.549 | 74.196 | 83.696 | 44.544 |
| Drifting action head | 13.608 | 16.383 | 16.545 | 13.943 |
| Action postprocessor | 0.316 | 0.414 | 0.437 | 0.320 |
| **端到端 action chunk** | **81.033** | **116.262** | **123.771** | **83.486** |

模型生成一个 chunk 的 P50 约为：

$$
T_{\text{model}} \approx T_{\text{VLM}} + T_{\text{Drifting}}
= 42.549 + 13.608
= 56.157\ \text{ms}.
$$

按 P50 计算，VLM 占 VLM 加 action head 时间的约 `75.8%`，Drifting head 占约 `24.2%`。因此继续减少 Drifting 的网络求值次数不会消除主要延迟，后续优化应优先关注 VLM、图像 token 数量和执行调度。

## Sync Action Queue

Checkpoint 配置为 `chunk_size=40`、`n_action_steps=40`。实际 `select_action` 测试结果：

| 操作 | 延迟 |
| --- | ---: |
| 第一次生成新 chunk | 62.917 ms |
| 39 次 queue hit P50 | 1.361 ms |
| 39 次 queue hit P99 | 1.727 ms |
| 39 次 queue hit 最大值 | 1.899 ms |
| 第 41 步重新生成 chunk | 58.958 ms |

30 FPS 的单周期预算为：

$$
T_{30\text{ FPS}} = \frac{1000}{30} \approx 33.3\ \text{ms}.
$$

Queue hit 明显低于预算，但新 chunk 生成约 `59-63 ms`。在同步 rollout 中，预计每执行完 40 个动作，也就是大约每

$$
\frac{40}{30} \approx 1.33\ \text{s},
$$

控制线程会因生成下一个 chunk 超出一帧预算。这与周期性推理停顿的现象一致。

## 视频 Pipeline 判断

Warm-cache 的三路本地 AV1 解码 P50 为 `6.977 ms`，P99 为 `8.001 ms`，不是本次离线全链路的主要瓶颈。之前跨随机 episode 的 cold-cache 测试出现过更高的首次打开延迟，但在线 Tron2 rollout 从 bridge 接收实时图像，不会读取这些 MP4，也不经过 TorchCodec。

因此，本测试只能排除 warm-cache 本地视频解码是主要瓶颈，不能排除在线 bridge 的以下问题：

- 相机帧与机器人状态对齐等待。
- 网络收帧或 JPEG/图像转换抖动。
- `get_latest_obs` timeout。
- bridge 线程和控制线程之间的锁竞争。

## 建议

1. 优先实现不带 RTC overlap guidance 的异步 chunk 预取。在当前 chunk 剩余一定步数时后台启动下一次推理，避免控制线程等待约 `60 ms`。
2. 在线日志分别记录 observation/bridge、preprocessor、VLM、Drifting head 和 action send 的 P50/P95/P99，并附带 action queue 剩余长度。
3. 如果继续优化模型延迟，先优化 VLM。可以评估减少图像 token、降低输入分辨率、减少相机数量、选择更早的 backbone layer，以及 `torch.compile`/CUDA Graph 等方案；这些更改需要重新验证策略成功率。
4. 不建议仅为消除同步停顿而实现完整 RTC。Drifting 当前不支持 action-prefix inpainting，而简单异步预取已能解决控制线程等待问题。只有新旧 chunk 的切换连续性仍然明显影响控制质量时，再评估 blending 或原生 RTC 训练。

## 限制与复现

- 这是 30 次 warm-cache 离线测量。P99 由小样本插值得到，接近最大值，不能作为生产环境的稳定 SLO。
- 本测试没有连接机器人，不包含 bridge、网络、状态同步和机器人 action send 延迟。
- 固定三个 episode 有利于稳定比较组件耗时，但不能覆盖 299 个 episode 的视频 cold-cache 行为。
- 堆叠图由 `scripts/plot_0724_inference_latency.py` 生成。运行前先执行：

```bash
source .venv/bin/activate
python scripts/plot_0724_inference_latency.py
```
