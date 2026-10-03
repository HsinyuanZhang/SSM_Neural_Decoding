# SSM 第二轮调试：容量、迁移和上下文

两张 RTX 3090 已完成 **52 个训练对照**：8 个旧架构容量控制、44 个现代模型和协议诊断。原始骨架确实过于简单，但当前低 R² 不能归结为宽度不足。评分只取 trial 尾部、跨 session 的输入/读出适配，以及训练和推理的上下文设置，都有更直接的实验依据。

所有新训练使用 seed 0。M1/M2 各保留一个本地 held-in session 作开发 target，分别为 `20120928` 和 `2020-10-28-Run1`；没有访问 held-out。这些 query 已用于研究诊断，不是盲测，也不是官方 FALCON/POSSM 复现结果。

## 1. 首先修正 R² 的解释

旧评分只使用长度 50 窗口的末端，丢弃每个 query trial 的前 49 bins。M2 trial 的中位长度约 57 bins，旧评分主要取运动尾部：2,747 / 14,115 个有效 bin，覆盖率 **19.46%**。M1 为 31,971 / 50,591，覆盖率 63.20%。

下面保持旧 checkpoint、prefix-33 ridge 和 query trial 分配不变，仅改变评分 bin：

| 任务 | 旧模型 | 旧尾部 R² | 同一批 query trial 全部有效 bin R² |
|---|---|---:|---:|
| M1 | Oscillator-64 | 0.4624 | 0.5172 |
| M1 | GRU | 0.4423 | 0.5136 |
| M2 | Oscillator-64 | -0.8698 | 0.1879 |
| M2 | GRU | -0.1525 | 0.1309 |

这是**评分样本变化**，不能称为网络提分。M2 旧尾部两个速度输出标准差约 0.00331/0.00310，全部有效 bin 为 0.01164/0.01025；尾部低方差使 R² 对预测误差更敏感。只凭负 R² 判断 SSM 无法学习运动，证据不足。

`all_valid_trial` 精确定义为：与旧实验相同、长度至少 50 bins 的 query trial，取其中全部 `eval_mask=True` 的 bin。短于 50 bins 的 trial 仍不在 target query cohort 内；`legacy_tail` 是其子集。训练则保留短于 context-128 的 source trial，通过左侧 padding 和 loss mask 处理。

R² 恢复物理单位后计算：`1 - sum((y-p)^2) / sum((y-mean_per_output(y))^2)`，即合并输出的 variance-weighted R²。采样证据见 [debug_audit.json](../results/debug_audit/debug_audit.json)。

## 2. 单纯加宽没有解决旧模型问题

原始 oscillator 只有单层线性输入、线性读出和简单递推，M1/M2 width-64 约 5.3k/6.5k 参数。旧 `selective` 仅实现简单 gate，不能视为完整 Mamba。新骨架加入了多尺度状态、残差、归一化、门控、FFN 和多层 SSM。

先保持旧训练/评分协议，只增加 oscillator 宽度。所有行使用 seed 0、600 updates、prefix-33 ridge、旧尾部评分：

| Width | M1 参数量 | M1 R² | M2 参数量 | M2 R² |
|---:|---:|---:|---:|---:|
| 64 | 5,328 | 0.4624 | 6,466 | -0.8698 |
| 128 | 10,640 | 0.4423 | 12,930 | -0.8188 |
| 256 | 21,264 | 0.4352 | 25,858 | -0.7897 |
| 512 | 42,512 | 0.4577 | 51,714 | -0.8872 |

width-256 训练至 2,000 updates 后，M1/M2 为 0.4513/-0.9116。宽度和更新次数本身没有解决旧 oscillator 的问题；这个控制不能否定更有表达力的架构。

## 3. 现代 SSM 与官方 Mamba

以下统一为 source-session 训练、source validation 选 checkpoint、target prefix-33 residual ridge，query 使用同一批 trial 的全部有效 bin。所有行使用 **trial 内 context-128**，与 recording-continuous 设置分别解释。

| Backbone | Width × layers / state size | M1 参数量 | M1 source-val / target R² | M2 参数量 | M2 source-val / target R² |
|---|---|---:|---:|---:|---:|
| Oscillator | 64 × 1 | 5,328 | 0.6417 / 0.5007 | 6,466 | 0.2911 / 0.1932 |
| GRU | 128 × 1 | 109,456 | 0.7787 / 0.4657 | 111,746 | 0.4677 / 0.1573 |
| S4D-style | 64 × 1 / 16 | 38,864 | 0.7460 / 0.5705 | 40,002 | 0.4544 / 0.2610 |
| S4D-style | 128 × 2 / 16 | 259,472 | 0.7859 / 0.5670 | 261,762 | 0.4805 / 0.0943 |
| S4D-style | 384 × 4 / 16 | 4,277,392 | 0.7716 / 0.5927 | 4,284,290 | 0.4472 / 0.0999 |
| Official Mamba-2 | 256 × 4 / 32 | 2,737,264 | 0.7789 / 0.5542 | 2,741,858 | 0.4965 / 0.2535 |
| Official Mamba-3 SISO | 128 × 2 / 16 | 352,736 | 0.7733 / 0.5073 | 355,026 | 0.5021 / 0.1184 |
| Official Mamba-3 SISO | 256 × 4 / 32 | 2,750,544 | 0.7847 / 0.4830 | 2,755,138 | 0.5334 / 0.1209 |

表达能力提升明显改善 source 拟合，target 却不随容量单调改善。M1 的大 S4D 有收益；M2 的约 4 万参数 S4D 已优于这组更大的 Mamba-3。因此目前不宜直接将默认网络扩大到数百万参数。

自研 S4D-style 使用稳定复极点、ZOH、多尺度状态、D skip、pre-LN residual、门控和 FFN，不是 POSSM 原实现。自研 SSD reference 与官方 Mamba-2 分开记录。官方 Mamba-2/Mamba-3 固定 upstream commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`，官方 core 外包裹 residual/FFN/readout；Mamba-3 只测 SISO，未测 MIMO。

Mamba-2 的 SSD、Mamba-3 更丰富的递推/复状态/MIMO值得保留，但语言任务收益不能替代本任务对照。[Mamba-2 paper](https://arxiv.org/abs/2405.21060)、[Mamba-3 paper](https://arxiv.org/abs/2603.15569)、[official source](https://github.com/state-spaces/mamba)。

## 4. 输入/读出适配有更直接的收益

加载 S4D-128×2 source checkpoint，沿用 source normalizer，冻结 SSM/FFN，仅训练输入投影、最终归一化和读出。target 前 33 个可用 trial 按 26/7 划分训练/验证，执行 500 updates；query 与原 cross-session 完全相同。

| 任务 | 适配前 target R² | IO 适配后 target R² | 可训练参数 / 总参数 |
|---|---:|---:|---:|
| M1 | 0.5670 | 0.6012 | 10,640 / 259,472 |
| M2 | 0.0943 | 0.2950 | 12,930 / 261,762 |

均为 all-valid、prefix ridge 分数。较少参数的适配就能改善迁移，为 session adapter 提供了直接依据；它不证明漂移来自某个特定生物机制。

另有 target-session 60% train / 20% validation / 20% query 学习能力诊断：

| Backbone | M1 query R² | M2 query R² |
|---|---:|---:|
| GRU-128 | 0.7535 | 0.3906 |
| S4D-128×2 | 0.7406 | 0.3424 |
| Official Mamba-3-128×2 | 0.7407 | 0.4486 |

这组使用**不同 query trial cohort**，只说明模型在较多目标 session 标签下能学习有效映射，不能与 cross-session 表直接排名。普通 prefix-from-scratch 和 IO warmstart 沿用原 query cohort，仅拟合协议不同。

## 5. Trial 边界和 SSM 状态

APST/SPINT 神经窗口在 recording/session 内连续，普通 trial boundary 不清空历史；当前主 SSM runner 的窗口在 trial 起点截断。两者都可以严格因果，但输入历史不同。

新增 `context_experiment.py` 实现 `recording_causal_fixed_window`：context-128 可读取前一 trial 的过去 neural bins，仅 recording 起点 padding；训练/验证 endpoint 和 loss mask 仍限定于各自 source bounds，query behavior label 不参与拟合。这是连续**输入窗口**，不是无限长度 persistent-state 训练，窗口自身仍从零初态计算。

对已有 checkpoint 只替换推理窗口，不重训，再与匹配新窗口重新训练比较：

| 任务 / S4D | 原 trial 窗口 R² | 固定权重改 recording 窗口 R² | recording 窗口重新训练 R² |
|---|---:|---:|---:|
| M1 / 64×1 | 0.5705 | 0.5648 | 0.5630 |
| M1 / 128×2 | 0.5670 | 0.5514 | 0.5714 |
| M2 / 64×1 | 0.2610 | 0.1927 | 0.2068 |
| M2 / 128×2 | 0.0943 | 0.1172 | 0.2666 |

均为同一 query cohort 的 all-valid ridge。M2-128 重新训练有明显变化，M2-64 则下降；连续上下文不是通用提分修补，也不应只改推理而忽略训练匹配。证据见 [context_only_audit.json](../results/debug_round2/session_context/context_only_audit.json)。

## 6. 骨架、验证和后续路线

模型为 [modern_models.py](../ssm_decode/modern_models.py)、[mamba2_official.py](../ssm_decode/mamba2_official.py)、[mamba3_official.py](../ssm_decode/mamba3_official.py)。训练为 [debug_experiment.py](../ssm_decode/debug_experiment.py)，连续窗口为 [context_experiment.py](../ssm_decode/context_experiment.py)。每个正式 fit 保存 best/last checkpoint、normalizer、曲线、split manifest、query 预测 NPZ 和物理单位 metrics。核查确认 44 个现代 fit 的这些文件齐全；其中 21 个有独立启动代码 hash 快照，23 个早期运行只有 manifest 中的代码 hash，不能追溯宣称全部记录了启动快照。见 [verification.json](../results/analysis/debug_round2/verification.json)。

source 按完整 trial 前 80% / 后 20% 划分训练/验证，normalizer 只拟合 source train。随机采样有效 endpoint，context-128、batch-32、最后半窗口有效 bin loss、AdamW、gradient clipping；验证固定最多 1,024 个 endpoint。best checkpoint 只由对应 validation 选择，没有使用 query 选 checkpoint。模型比较本身使用本地开发 query，仍需后续跨 target、跨 seed 检验。

最终合并运行相关核心/runner/data/官方 Mamba-2 CPU constructor/上下文测试，**37 passed in 1.96s**。其中 4 项上下文契约测试覆盖过去跨 trial 输入、未来扰动因果性、训练 bounds 外标签扰动不改变 masked numeric loss，以及旧/新 support/query 索引相同。官方 Mamba-2/Mamba-3另有真实 CUDA forward/backward/因果性 receipt。不声称已经运行整个仓库 suite。

环境为 `PYTHONNOUSERSITE=1`、Conda `spint` Python 3.10 / PyTorch 2.5.1；官方模型使用 Triton 3.5.0 隔离目录 `.tools/mamba_deps` 和固定源码 clone，未升级主环境。一次 Mamba-3 CUDA 失败保留记录；设置 default CUDA device、单卡可见性及独立 cache 后已完成正式运行，具体 cache 根因未被证明。早期四个 context fit 的 adapter hash 是训练后 receipt；后续入口已增加启动期 hash，不回写历史启动快照。

下一阶段优先保留两条可检验路线：

1. **S4D-64/128 + session IO adapter**：已有直接跨 session 证据，固定 frontend、query cohort 和上下文，再独立扫 layers/state size。
2. **APST frontend + 官方 Mamba-2/Mamba-3**：稳定接口是 `[B,T,256] -> temporal backbone -> [B,T,256]`。保留 frontend、bank、normalizer、objective/readout，单独替换 temporal backbone，才能回答最新 SSM 对现有系统是否有净收益。

POSSM 使用逐 spike tokenization/cross-attention 与 recurrent SSM 的混合结构；当前 raw linear frontend 不具备相同表征，不能只比较 backbone 或参数量就声称超过 POSSM。[POSSM paper](https://arxiv.org/abs/2506.05320)。

## 7. 复现和图表

从 SSM 目录重跑连续窗口对照，输出必须是新目录：

```bash
cd /home/xinyuan/Work_host/SSM
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python -m ssm_decode.context_experiment \
  --task m2 --device cuda:1 --width 128 --layers 2 --state-size 16 \
  --context 128 --steps 2000 --output results/replay_m2_context128
```

两 GPU 矩阵为 `configs/debug_round2.json`、`configs/debug_round2_latest.json`、`configs/debug_round2_diagnostics.json`；官方 Mamba-2 矩阵另存 `results/debug_round2/official_m2_matrix.json`。新运行与聚合命令：

```bash
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python scripts/run_debug_two_gpu.py \
  --matrix configs/debug_round2_latest.json --output-root results/replay_latest
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python -m ssm_decode.debug_report \
  --results results --output results/analysis/debug_round2
```

[summary.csv](../results/analysis/debug_round2/summary.csv) 含 340 条标准化 score 行，按 `mode`、`context_policy`、`score_scope`、`adaptation` 过滤。44 个现代 fit 和 8 个 width-only fit 是训练计数；CPU linear、sampling/context audits、smoke/失败记录不计正式 fit。

PNG/PDF 图表：[容量控制](../results/analysis/debug_round2/scale_vs_r2.png)、[参数量与 target R²](../results/analysis/debug_round2/params_vs_allvalid.png)、[source validation 与 target R²](../results/analysis/debug_round2/sourceval_vs_target.png)、[source 学习曲线](../results/analysis/debug_round2/traincurves.png)。参数和迁移图仅取 cross-session、all-valid、ridge，区分 trial/recording context。

## 8. 下一轮 cross-session：正式比较 SSM 微调方法

2026-10-03 文献和代码核查后的路线修订：association-profile 保留为输入表征选项与对照，cross-session 适配不再限定为 profile/IO/ridge。Mamba-3 SISO 保留为主要候选，与 Mamba-2、S4D 对照；首先回答可迁移的 source dynamics 在新 session 中需要改动哪一部分，再决定最终 backbone。

SISO 指内部递推形式，不意味着只接受一个电极或只输出一个动作维度。当前 decoder 已处理 M1 的 64 输入/16 输出和 M2 的 96 输入/2 输出。Mamba-3 的复状态、更丰富的离散递推与选择性具有建模潜力，但它本身不提供 neural channel identity 的跨 session 对齐。论文验证主要是语言、检索和状态追踪，而非本项目的跨 session 解码。[Mamba-3](https://arxiv.org/html/2603.15569v1)。

已有 Mamba-3 cross-session 数字来自 source training + prefix residual ridge，没有正式进行 Mamba-3 backbone PEFT；target-prefix from-scratch 也不等于 source checkpoint 的微调。因此目前的低 target R² 不能用于否定其 PEFT 能力。

POSSM v2 Appendix D.9 已有 LoRA 对照：o-POSSM-GRU 的 embedding LoRA 在六个 T–RT held-out sessions 上为 0.7478±0.0634，UI 为 0.7464±0.0692；平均可训练参数 9.16K/15.6K。这是 embedding/GRU 版本的初步验证，不是 Mamba-3 dynamics-LoRA，也说明“新 backbone + 普通 LoRA”本身不能作为超过 POSSM 的证据。[POSSM D.9](https://arxiv.org/html/2506.05320v2#A4.SS9)。

| 方法 | 核查的原始来源 | 下一轮定位 |
|---|---|---|
| Projection LoRA、逐步解冻 full FT | [POSSM](https://arxiv.org/html/2506.05320v2)、[MambaPEFT, ICLR 2025](https://arxiv.org/abs/2411.03855) | 必需基线；判断更复杂 SSM 特定适配是否值得 |
| Sparse Dimension Tuning + projection LoRA（SDLoRA） | [Parameter-Efficient Fine-Tuning of State Space Models, ICML 2025](https://arxiv.org/abs/2410.09016)、[作者代码](https://github.com/furiosa-ai/ssm-peft) | 优先的 SSM 特定候选；移植到 Mamba-3 是新的实现与实验工作 |
| State-offset tuning | [ACL 2025](https://arxiv.org/abs/2503.03499)、[作者代码](https://github.com/furiosa-ai/ssm-state-tuning) | 状态适配对照；需遵守真正的 recurrence/state 语义，不能把普通 block-output bias 冒称 state-offset |
| Memba：LoRA + membrane gating/transfer | [ICLR 2026，2026-03-02 修订](https://arxiv.org/abs/2506.18184)、[作者代码](https://github.com/Intelligent-Computing-Lab-Yale/Memba) | 较新探索项；先核查 Mamba-3 兼容性，再评估额外时序状态是否带来净收益 |

这些 SSM PEFT 工作主要验证语言/视觉模型。已核查来源不能证明它们在 Mamba-3 intracortical cross-session 上有效，也不能据此宣称某个方法已成为该任务的通用主流。LoRA/full FT 是成熟基线，SDT/state-offset/Memba 是有来源、值得验证的 SSM 特定方法。

### Mamba-3 参数必须按实际结构选择

官方 `blocks.i.ssm.in_proj` 同时生成 `[z, x, B, C, dd_dt, dd_A, trap, angles]`。普通整矩阵 LoRA 会同时改变 gate、value、状态读写、时间尺度、衰减、积分系数和旋转，不能称为仅输入适配。`dd_A` 经负值参数化，`DT=softplus(dd_dt+dt_bias)`；Mamba-3 并没有可直接照搬 Mamba-1 调参代码的独立 `A_log`。[本地固定版本](../.tools/mamba_official/mamba_ssm/modules/mamba3.py)。

首轮参数分组如下，均为待检验方案：

1. 投影 LoRA/partial tuning：frontend 或输入投影、SSM output projection、norm/readout，冻结产生衰减和旋转的投影参数，建立可复现基线。
2. 结构化读写适配：针对 B/C 对应行的低秩更新、B/C bias/norm；保持其余 core 参数冻结，与普通整矩阵 LoRA 比较。
3. 有限时间尺度适配：在第二组基础上独立加入 `dt_bias`，较小学习率和相对 source 参数的约束；是否需要 `dd_A`/angles 的变化交给后续消融。
4. SDT/state-offset/Memba：各自按原方法实现，记录与 Mamba-3 原结构不兼容的部分，不把上述自定义分组直接命名为原论文方法。

当前 wrapper 未暴露官方 `inference_params/step`，仍是固定窗口重算。state adapter 之前需补齐状态 API，并验证 angle/state/previous-K/previous-V 的分块与逐步一致性。SSM 隐状态在独立 recording/session 开头重置；不能用携带前一独立 session 的缓存代替适配。

### 比较合同与研究问题

在相同 frontend、source checkpoint、support labels、query cohort 和 context 下，比较：IO-only、普通 LoRA、SSM 特定适配、逐步解冻 full FT；AP 开/关作为独立 frontend 因素。新增方法首先与同 backbone 的 IO-only 比较，再比较 Mamba-2/Mamba-3/S4D，避免同时改变多个因素。

训练侧继续使用多 session 神经数据预训练；不用语言预训练权重替代 neural pretraining。目标侧先采用已有 labelled prefix 的监督 PEFT，并报告实际可训练参数、适配耗时和 R²。无标签在线适配另设协议，只能访问已观测神经数据，不能通过预测熵等分类目标直接套用连续回归。

候选研究问题是：跨 session 变化能否由少量状态读写与时间尺度参数吸收，同时保留 source 预训练的共享动态？先在固定本地 target 筛选，再在多个 development target、多个 seed 和不同 support 数量上确认；query 不参与优化、早停或稀疏维度选择。

本节交付的是证据核查和下一轮方案，尚未运行上述新 PEFT 方法，不能将第二轮已完成的 52 个 fit 当作它们的验证结果。
