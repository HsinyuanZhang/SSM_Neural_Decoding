# Round1 本地 pilot 结果（2026-10-03）

完整来源为 `results/round1/m1` 与 `results/round1/m2`：每项都有 `run.json: completed`、15 个完整 checkpoint（5 kinds × 3 seeds）和 per-session JSONL。汇总及逐 seed 图见 [results/analysis/round1/report.md](../results/analysis/round1/report.md)。这不是 POSSM 复现、官方 evaluation 或任何 superiority claim。

## 协议与可用范围

- M1 为 16 维 EMG，M2 为 2 维 finger velocity；两者均为固定通道 binned-feature pilot，训练为 600 steps，不是 CLI 默认的 1200 steps。
- 每个 kind/adaptation/budget 报告 physical variance-weighted `R²` 的三 seed mean ± sample SD。target 是协议指定的 held-in local-only 最新 session；该 session 不进入 source normalizer/source set。所有 minival targets 因只剩 2 个可用 trials 而被 skip，skip receipts 已保存，未产生分数。
- 相同 target 内的 k=5/10/33 使用 `common_query_tail`（cutoff=33），故 calibration curve 的横向预算可比较。M1 session 为 `20120928`，M2 session 为 `2020-10-28-Run1`。

## exact RLS（ridge 与此实现数值相同）的主要表

| Task | k | diag | osc | bank | selective | GRU |
|---|---:|---:|---:|---:|---:|---:|
| M1 EMG16 | 5 | 0.1503 ± 0.0898 | 0.3258 ± 0.0421 | 0.2031 ± 0.0275 | 0.1222 ± 0.0712 | 0.3510 ± 0.0388 |
| M1 EMG16 | 10 | 0.2261 ± 0.0309 | 0.3870 ± 0.0255 | 0.2933 ± 0.0609 | 0.1828 ± 0.0500 | 0.3798 ± 0.0322 |
| M1 EMG16 | 33 | 0.3946 ± 0.0229 | 0.4595 ± 0.0054 | 0.4283 ± 0.0165 | 0.3775 ± 0.0305 | 0.4223 ± 0.0178 |
| M2 finger velocity2 | 5 | -1.4936 ± 0.7966 | -0.6006 ± 0.0440 | -0.7616 ± 0.2851 | -0.9844 ± 0.7884 | -0.1769 ± 0.0878 |
| M2 finger velocity2 | 10 | -1.8683 ± 0.8895 | -1.0315 ± 0.1084 | -1.6512 ± 0.3410 | -1.1968 ± 0.9141 | -0.2219 ± 0.0557 |
| M2 finger velocity2 | 33 | -0.9419 ± 0.2127 | -0.8720 ± 0.0370 | -1.1474 ± 0.1017 | -0.7902 ± 0.4547 | -0.1859 ± 0.0314 |

## 结论与限制

M1 上 bank 在 k=33 高于 diag，但低于 osc；M2 上所有列的 mean `R²` 都为负，GRU 相对最高。因而本轮不支持“当前固定 grouped bank 优于 diag/osc/selective”的结论，也不支持跨任务收益。更严格地说，osc/bank state 形状是 `[width,2]`，即 width=64 时有 128 个 recurrence state values；diag/selective/GRU 为 64，当前没有 state-matched 的 width=128 diag/selective/GRU 对照。因此 bank 与这些模型的直接比较还不满足 2× state fairness，不能据此归因给振荡或 codebook。

## Profile phase2（独立条件，不能与 raw 混合）

`results/profile_round1/m1` 和 `m2` 均已完成 3 kinds × 3 seeds，使用 `support_normalize_neural=false`，且 `metrics.json` 将 `profile` 与 deterministic `profile_drop20` 分开。它学习 profile frontend 的权重，训练输入/参数化不同于 raw pilot，因此下列值不能当作 raw 与 profile 的 paired architecture comparison。

| Task | condition / adaptation | diag | osc | bank |
|---|---|---:|---:|---:|
| M1 | profile ridge, k=33 | 0.3438 ± 0.0463 | 0.3432 ± 0.0093 | 0.3750 ± 0.0133 |
| M1 | profile_drop20 zero, k=33 | -0.9763 ± 0.5087 | -0.7811 ± 0.5490 | -0.3793 ± 0.3121 |
| M2 | profile ridge, k=33 | -0.1479 ± 0.0370 | -0.3544 ± 0.0595 | -0.1699 ± 0.0357 |
| M2 | profile_drop20 zero, k=33 | -114.9902 ± 50.8290 | -34.5098 ± 7.4568 | -59.6672 ± 68.4612 |

M1 的 profile ridge 下 bank 是这三个 profile kinds 中最高者，但其值低于 raw osc ridge 的 0.4595；M2 profile ridge 仍为负。尤其是 profile zero 与 drop20 出现极大的负值和种子方差，说明当前 learned profile frontend 的无校准输出不稳定，不能据此宣称 profile 抵抗 channel dropout 或跨 session drift。该失败现象保留“神经/跨 session shift 是主要瓶颈”的假设，但不识别其原因，更不支持 neural/chip joint compensation。

## Support-only neural-normalized profile phase3（再次独立）

phase3 仅把 neural normalization 从每个 target/source session 的 support prefix 拟合，query labels 不参与此拟合；其 `support_normalize_neural=true`，所以它是 `profile_normalized` family，绝不可与 plain profile 或 raw 合并。其 profile-ridge、k=33 为：

| Task | diag | osc | bank |
|---|---:|---:|---:|
| M1 | 0.3769 ± 0.0265 | 0.3826 ± 0.0036 | 0.4044 ± 0.0142 |
| M2 | -0.8950 ± 0.2656 | -1.0779 ± 0.0325 | -1.2399 ± 0.1400 |

M1 的 normalized bank 高于 plain profile bank（0.3750），但仍低于 raw osc（0.4595）；M2 的 normalized ridge 反而低于 plain profile 的 diag（-0.1479）和 bank（-0.1699）。因此现有证据拒绝“support-only neural normalization 解决 M2 severe mean shift”的假设。原因仍未识别：可能是支持估计误差、source/target 特征不匹配或模型训练问题；不能把它归因给芯片误差，也不能从一个 target session 推广。

## Direct/readout baseline、M2 方差与 folded export

固定 `alpha=1`、相同 k=33 query tail 的 direct neural ridge baseline 在 M1 的 lag-0 为 0.3181、lag `[0,1,2,4]` 为 0.3709；M2 分别为 -0.9744 和 -2.3350。frozen latent64 residual ridge 的 M1 osc 为 0.4488，M2 GRU 为 -0.9576。baseline JSON 见 `results/round1/readout_baselines.json`，它说明当前 M2 困难不是仅由某一个 SSM bank 造成，但也没有给出充分的因果诊断。

development-only 诊断中，M2 query behaviour physical std 是 support 的约 0.31/0.27；低方差 query tail 会使 `R²` 对绝对误差更敏感。因此 M2 的负分数不能被称为“灾难性 neural drift”。query labels 没有用于任何 fit 或模型选择；它们只用于此解释性诊断。对 neural mean shift 的判断仍需更多 target session 与预注册诊断。

四个 profile family/task 的 folded projection export 共 36 份，first-50-bin normalized-vs-folded 最大误差不超过 `5.7220e-6`。这是 raw projection 与 folded FP32/affine/int8 materialization 的数值等价证据，不是 accuracy 或芯片性能结果。projection state 标记为 `mutable_per_session_profile_generated`，因而应计入 session calibration 的 SRAM 写入/存储；不能把它过度描述成固定 RRAM 权重。

## Checkpoint characterization 与 fake-quant robustness

30 个 raw checkpoint 的 CPU characterization 已写入 `results/round1/characterization.json`；由 k=33 raw ridge mean 和 seed-0 resource record 组成的表在 `results/analysis/round1/resource_accuracy.csv`。M1 的 diag/osc/bank/selective estimated MACs per step 为 5184/5376/5376/5248，state bytes 为 256/512/512/256；M1 GRU 为 29696 MACs 和 256 state bytes。M2 对应为 6336/6528/6528/6400 与 256/512/512/256；M2 GRU 为 30848 MACs 和 256 state bytes。它们是代码路径的算术/存储 accounting，不是 FPGA、ASIC 或 CIM 的能耗、面积或时延测量。

fake-quant robustness 有 60 个 seed-0 rows，W8 权重和 state scale 都遵守“target support 标定、query 冻结”。M1 osc 的 ridge `R²` 从 clean float-state 0.4624 到 W8 float-state 0.4625、W8 S8 0.4605、W8 S16 0.4645；bank 对应 0.4294、0.4291、0.4266、0.4287。该单 target/seed 的小变化值得作为下一轮整数实现的检查点，但不能成为 hardware-friendly、量化无损或跨模型保证。S8/S16 query state 的实际 clipped component counts 对 osc 为 73/77、bank 为 117/120；因此也不能声称 S16 没有 clipping。图见 `quantized_r2.{png,pdf}` 和 `resource_accuracy_pareto.{png,pdf}`。

## 基于现有证据的下一步排序

1. 保留 raw osc + ridge/RLS 作为 M1 的局部开发基线，并先补 state-matched（width=128）对照，再讨论 oscillator/bank 的归因。
2. M2 优先做诊断而不是继续扩展模型：按 source/target 的均值、方差、缺失 channel、support length 和 query tail 分解失败；不要把 negative `R²` 伪装成校准收益。
3. learned shared bank、irregular sampling、meta-Delta 与 neural/chip drift joint compensation 维持候选状态，直到多 target session、state-matched 和有效的 profile/dropout 条件支持它们。

所有这些判断只来自每任务一个 held-in local target，POSSM 的准确性、速度、官方分数和跨物种结论均未在此验证。

Delta 的所有预算和 zero/ridge/RLS 细表均在 `r2_summary.csv`；只应以其三 seed 分布解释。learned shared bank、irregular sampling、meta-Delta，以及 neural/chip drift joint compensation 仍是未实现假设。

数字 FPGA/ASIC 与 RRAM-CIM 路线保留同一 integer 递推接口；RTL 已对 256 条 Python 向量逐拍通过，但这不提供能耗、面积、ADC/DAC、器件漂移或芯片性能数据。
