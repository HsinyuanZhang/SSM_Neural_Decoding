# Cross-session SSM 迭代记录（2026-10-03）

本轮依据 `ssm_cross_session_audit_20261003.md` 推进。当前状态：正式 A 矩阵运行中，B 阶段实现与 CPU 验证完成，最终性能尚未验收。仓库 README 保持简短。本轮不增加 SystemVerilog 工作。

## 研究合同与实现

A 阶段固定已有 Mamba-3 SISO source checkpoint（256×4、state 32），先修正 target 输入合同，再比较适配参数化。target 仅使用前 33 个长度 ≥50 bin 的 trial 做 support。33 个 trial 的无标签神经数据用于逐通道 z-score，source 的输出归一化统计保持不变。监督适配使用 26 个训练 trial 和 7 个交错验证 trial，索引为 4/9/14/19/24/29/32（零起点）。query 标签在拟合数据中被替换为 NaN。

A 的方法为 none、IO、projection LoRA、channel affine、affine+LoRA、rotated state-offset、original-coordinate state-offset、逐步解冻的 full tuning。LoRA 允许训练根层输入 bias。两种 offset 使用同一 affine、同一低秩规模、同一最终 BF16 量化，仅改变 readout 坐标，不附加根层 LoRA。channel affine 可代数折叠到输入线性层。

学习率按方法族扫：IO 为 3e-4/1e-3，LoRA、affine 与 offset 为 3e-4/1e-3/3e-3，full 为 1e-4/3e-4。初始预算 2,000 步，若 best_step ≥0.8×预算则延长，最大 8,000 步。到最大预算仍未满足条件的候选不能参与 LR 选择。同一 LR 必须三个 seed 全部满足条件。按三个 seed 的 prefix-val 均值选 LR，随后仅评估选中的 query checkpoint。none 为一个确定性重复。

矩阵共 116 个训练候选、44 个选中模型的 query evaluation。GPU 0 运行 M1，GPU 1 运行 M2。正式根目录为 `results/cross_session_iteration_a/official_20261003`。主指标是 query trial 的全部 eval-valid bin 上的物理单位 variance-weighted R²。direct 为主指标，ridge 为辅助，M2 legacy tail 只作诊断。

每个运行保存冻结 matrix、源代码、source checkpoint/normalizer hash、target 文件 hash、官方 kernel runtime、best/last checkpoint hash、train/val/query bounds、每一步 loss/梯度、验证曲线、参数变更和冻结审计。复用已有结果必须同时匹配确定路径、task/method/seed/lr、matrix 参数和这些哈希。query selection 可从 prefix 产物独立重建。

B 的 source 训练实现包含逐 source-session 的 train-only 输入统计、共享骨架和逐 session gain/bias/latent embedding、跨 trial 边界的 recording 窗口、训练标签分区掩码、session-balanced sampling，以及逐通道漂移增强。target 从未被 source trainer 加载。导出时将 source 前端均值折叠到普通 backbone 的 `in_proj`。这些 affine 前端只能缩放共享权重的列，不能代替 POSSM 的完整 unit embedding 重学习，因此 full IO 仍是必要对照。

## 独立复核与原审核文档的两处修正

独立 reviewer 为用户指定的 `gpt-5.6-sol`、`xhigh`。训练前完成八种方法的官方 GPU kernel smoke：step zero 与 source 输出一致，未来输入扰动不改变已有 prefix，七种可训练方法均有非零有限梯度，冻结参数无变化。

1. 原审核 EMA 脚本用整个 support 初始化，再从 recording 起点更新，重复纳入了 support；其方差更新还使用更新后的均值。本轮将 prefix 冻结到最后一个 support trial 的结束位置。之后每个 neural bin 先输出再更新，方差使用更新前的 delta，公式为 `v'=(1-alpha)*(v+alpha*delta²)`。不使用 query 标签。半衰期固定为 3,000 bin，单独报告为严格因果无监督测试时适配，不能与 fixed-support 协议混成一个主表。
2. 官方 kernel 的 `R(+Phi)` readout 约定给出 `(R(Phi)q)^T h' = q^T R(-Phi)h'`。原文 F8 的物理坐标换算符号须修正。两个版本都有数学定义；rotated-frame 常量 offset 在原始坐标下随相位变化。original-coordinate 版本使用不旋转的 normalized C+bias。本轮不声明已经验证官方 `step()` 流式等价。

新预测 archive 保存原始物理标签，而非旧 round-1 的 float32 normalize/denormalize 往返标签。旧、新 truth hash 因此可能不同。验收要求新 archive 与当前 target 标签按索引逐元素一致，并要求同一任务的新方法具有完全相同的 index/truth hash。

## 已完成的无训练归一化重评

这些数值是固定旧 source 的独立诊断，尚不是正式 A 矩阵的适配结果。

| 协议 | M1 direct R² | M2 direct R² | 可训练参数 |
|---|---:|---:|---:|
| 旧 source 统计 | 0.3862094 | -0.0059217 | 0 |
| fixed support z-score | 0.6735682 | 0.2337932 | 0 |
| 修正的 strictly-past EMA，half-life 3000 | 0.6992543 | 0.2669104 | 0 |

float64 物理单位 oracle 与 GPU 评分的最大差异为 6.75e-8。最终正式表由保存的预测数组重算，同时核对原始标签、cohort 和训练身份。

## 复现入口

```bash
cd /home/xinyuan/Work_host/SSM
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  scripts/run_cross_session_iteration.py \
  --matrix configs/cross_session_iteration_a.json \
  --output-root results/cross_session_iteration_a/official_20261003 --phase all

PYTHONNOUSERSITE=1 PYTHONPATH="$PWD" /home/xinyuan/miniconda3/envs/spint/bin/python \
  scripts/summarize_cross_session_iteration.py \
  --root results/cross_session_iteration_a/official_20261003 \
  --output results/analysis/cross_session_iteration/stage_a --require-complete
```

正式结果、方法结论与 B 阶段证据将在独立验收完成后写入本记录。单 source seed、单 target 的本地 held-in 开发协议不能证明超过 POSSM，也不能直接比较 APST 的官方 held-out 成绩。
