# Unit/session 前端迭代：方法与执行协议（2026-10-04）

## 定位与边界

本迭代在不改变 Mamba-3 主干的前提下，比较会话专属的输入/输出坐标映射。这里的“unit”表示**记录会话内通道坐标**：不同会话的神经通道可以有不同增益、偏置和方向校正。它不主张、也不估计跨会话的生物神经元身份对应关系。

这次可行性研究已完成本地训练与评分，结果显示局部收益和适配退化并存。它不是 POSSM 复现，也不包含 attention 或 streaming 能力主张。旧 A/B 实验和 `README.md` 保持不变。后续只在匹配基线收益且在多个 target 上保持该收益后，再评估 unit-token attention。

## 模型与消融

对 batch 中会话 \(s\) 的归一化输入 \(x\in\mathbb{R}^{B\times T\times C}\)，共享主干输入投影为 \(W\in\mathbb{R}^{D\times C}\)、偏置为 \(b_0\)。前端使用：

\[
z = W(x\odot g_s+b_s)+e_s+x\Delta_s^\mathsf{T}.
\]

其中 \(g_s,b_s\in\mathbb{R}^{C}\)，\(e_s\in\mathbb{R}^{D}\)，\(\Delta_s\in\mathbb{R}^{D\times C}\)。方向残差项直接作用于原始归一化 \(x\)，不作用于 \(x\odot g_s\)。之后依次经过原有 `blocks`、`final_norm`、`out_proj`。若启用会话读出，输出为

\[
y_s=\exp(q_s)\odot y+r_s.
\]

配置 `configs/unit_session_iteration_20261004.json` 定义三个消融：

1. `affine`：仅输入通道仿射 \(g_s,b_s\) 与 latent 偏置 \(e_s\)。
2. `unit`：在 affine 基础上加入完整方向残差 \(\Delta_s\)。
3. `unit_readout`：在 `unit` 基础上加入读出对数增益 \(q_s\) 与偏置 \(r_s\)。

有效映射是正则化、跨会话比较和折叠的对象：

\[
\begin{aligned}
A_s &= W\operatorname{diag}(g_s)+\Delta_s,\\
c_s &= Wb_s+b_0+e_s,\\
V_s &= \operatorname{diag}(\exp(q_s))O,\\
d_s &= \exp(q_s)\odot o_0+r_s.
\end{aligned}
\]

未启用方向残差时 \(\Delta_s=0\)；未启用会话读出时 \(q_s=r_s=0\)。正则项只约束有效映射。令 \(M\in\{A,c,V,d\}\)，source 阶段的参考映射为 \(\bar M=\frac1S\sum_sM_s\)，target 阶段的参考映射为初始化锚点 \(M^{(0)}\)。两阶段均使用配置中的 \(\lambda=0.01\)：

\[
\mathcal P=\lambda\left[
\frac1{SC}\sum_{s,c}\sum_d(A_{sdc}-R_{sdc})^2
+\operatorname{mean}_{s,d}(c_{sd}-R_{c,sd})^2
+\operatorname{mean}_{s,o,d}(V_{sod}-R_{V,sod})^2
+\operatorname{mean}_{s,o}(d_{so}-R_{d,so})^2\right],
\]

其中 source 取 \(R=\bar M\)，target 取 \(R=M^{(0)}\)。这正对应 `mapping_penalty`：`input_weight` 先沿 latent 行 \(d\) 求和，再对 session 与 channel 平均；其余三项对所有元素取均方，四项相加后乘 \(\lambda\)。它对参数分解的 gauge 变换不变，例如 \(g_s\leftarrow g_s+\alpha,\ \Delta_s\leftarrow\Delta_s-W\operatorname{diag}(\alpha)\)，以及 \(b_s\leftarrow b_s+d,\ e_s\leftarrow e_s-Wd\)，不会改变 \(A_s,c_s\)。

source 到 target 的初始化取有效输入和归一化读出映射的会话均值。实现上输入参数的算术均值因共享 \(W\) 而精确给出均值 \(A,c\)；读出增益使用 \(q_*=\log\operatorname{mean}_s\exp(q_s)\)，读出偏置取 \(\operatorname{mean}_s r_s\)，从而折叠后的 \(V,d\) 等于逐会话折叠映射的均值。

## 数据、归一化与选择

source 训练只使用 held-in source 会话，source seed 固定为 0。每个 source recording 以内部 tail trial partition 作验证；source 验证按会话先计算 R²、再取宏平均，使长记录不会支配 checkpoint 选择。target 适配使用 public calibration 的原始 trial：M1 使用前 10 个，M2 使用前 33 个；每第五个 raw trial 留作验证，M1 为 `[4, 9]`，M2 为 `[4, 9, 14, 19, 24, 29, 32]`。这些短 trial 仍保留。leading neural-only 段参与 target 神经输入统计，但没有监督标签。

target 输入统计由全部合法 calibration prefix neural bins 计算；行为输出的均值和标准差固定为 source 训练统计，不能被 target calibration 行为改变。窗口中的非本分区行为标签设为 `NaN`，mask 为 false，因此不能参与训练或验证。

target 的 `none` 初始化只运行一次：它使用 calibration prefix 的输入归一化和 source 有效映射均值，不进行监督权重更新。可训练 `ui` 适配对每个学习率运行 3 个 target seeds（0、1、2），候选学习率为 `0.0003` 与 `0.003`。选择依据是三个已收敛 seed 的平均 prefix-validation R²。query 不用于 checkpoint、学习率或 seed 选择；在所有 fit 封存后，local query 用于本轮架构探索比较。训练预算从 2k 起；若最佳验证点位于预算后 20%，才扩至 4k，再至 8k。达到最大预算时仍出现 late maximum 的 fit 被排除。

query cohort 固定为 usable target trial 的第 34 个及以后、且仅取 eval-valid bins。canonical cohort 计数为 M1 50,591 bins、M2 14,115 bins；索引和物理标签均有配置内 SHA-256 绑定。

## 封存、折叠与评分

fit 阶段完成全部 source/target fits 后，分别写入每任务 `fits_complete.json`，再由 launcher 汇总为 `fit_seal.json`。评分前必须在外部记录 `fit_seal.json` 的 SHA-256；score 阶段重新核对该 seal、任务完成清单、contract 和全部拟合工件哈希。完成这些核对前不会读取 query 标签。

source 的折叠回放独立覆盖各 session 的完整 source validation。target 的每个候选部署模型则在全部合法 prefix、所有 eval-valid endpoints 上进行折叠回放，范围包含 raw train 与 validation partitions：逐点 unmerged/folded 输出必须 `allclose`，且物理空间 R² 差的绝对值不超过 0.001。任一条件失败时，该折叠模型不作为部署候选。

## 已完成的本地开发结果

本轮已完成 48 个 fit、24 个选中模型的 local query 评分和 55 个测试；分析器以外部 fit seal `d54522362911480a0289c0dd9f87357757cc318a3c5517d58b8de238f26d7c8b` 复算保存数组的物理 float64 R²、核对 canonical cohort、拟合工件、折叠收据与 archive 闭包。独立审计已完成训练前实现、拟合封存及最终评分检查，结果为 **PASS**。最终评分审计使用独立脚本复算全部 24 个 query archive，并确认六组 JSON/CSV 结果一致；见 [最终评分审计](../results/analysis/unit_session_iteration_20261004/review/final_score_audit.json)。

另从原始数据独立重放 M1 `unit/ui` seed 0（LR 0.003）与 M2 `unit_readout/ui` seed 0（LR 0.0003）的完整 query。两例索引、物理标签和全部预测均逐元素一致，预测最大绝对差与 R² 差均为 0；见 [GPU 重放审计](../results/analysis/unit_session_iteration_20261004/review/gpu_query_replay_audit.json)。重放没有改变正式预测或权重。

结果仅来自 local held-in development，不读取 `results/official_evalai`，不是官方 held-out 结果。query 没有用于 checkpoint、LR 或 seed 选择；它在所有 fit seal 后用于本轮架构探索比较。每个任务仅有一个 source seed（0）和一个 target；target 的三个 seed 及其 SD 不构成 cross-session 泛化证明，也不改变“会话通道坐标、不对应生物 unit identity”的解释边界。

| 任务 | 映射 | none R² | UI 三 seed 平均 ± population SD | 选中 LR | target 可训练参数 | source full-validation 宏平均 | 方向诊断 | 选择/排除 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| M1 | affine | 0.6878 | 0.7036 ± 0.0033 | 0.0003 | 256 | 0.7841 | \(4.05\times10^{-16}\)，affine 近零 | 选中；无 LR 排除 |
| M1 | unit | 0.7014 | 0.5803 ± 0.0059 | 0.003 | 8,448 | 0.7847 | 0.1106 | 选中；无 LR 排除 |
| M1 | unit_readout | 0.6998 | 0.6962 ± 0.0007 | 0.0003 | 8,480 | 0.7842 | 0.1091 | 选中；无 LR 排除 |
| M2 | affine | 0.3938 | 0.2432 ± 0.0718 | 0.003 | 448 | 0.6164 | \(5.20\times10^{-16}\)，affine 近零 | 选中；慢 LR family 排除（3 个 late） |
| M2 | unit | 0.3888 | 0.4115 ± 0.0014 | 0.0003 | 25,024 | 0.6004 | 0.1816 | 选中；无 LR 排除 |
| M2 | unit_readout | 0.3585 | 0.4210 ± 0.0020 | 0.0003 | 25,028 | 0.6030 | 0.1806 | 选中；无 LR 排除 |

方向诊断基于有效输入映射 \(A[S,D,C]\)：对每列将 \(A_s\) 投影到 source 均值方向，报告正交残差能量除以总 \(A\) 能量。它在 affine-only 中接近零符合构造，在带方向残差的条件中为约 0.11（M1）或 0.18（M2）；这是映射几何量，不是生物神经元关联证据。

M1 的 `unit` none 比 affine none 高 0.013575，但其 UI 均值降至 0.5803；`unit_readout` 的 0.6962 未超过 affine UI 的 0.7036。同一变体的 UI 相对 none 变化分别为 affine +0.015747、unit −0.121138、unit_readout −0.003518。M2 中 `unit` 和 `unit_readout` 的 UI 分别为 0.41145 与 0.42098，同一变体的 UI 相对 none 变化为 +0.022681 与 +0.062454；affine 则为 −0.150654，且其 UI SD 为 0.07182。完整 `unit_readout` UI 相对 affine none 的收益只有 +0.027169，不能用不稳定的 affine UI 夸大收益。当前 M2 结果也低于上一轮 wide-LoRA 的合法 M33 开发均值 0.482141；两者使用同一 query cohort，但训练协议不同，因此该数值仅是历史参考。

M1 source affine 的 session `20120927` 和 M2 已排除的慢 affine `lr=0.0003` family 中 seed 0 均未通过折叠逐点门槛；两者的 R² 差门槛仍通过。M2 的整个慢学习率族另外因三个 seed 在最大预算仍有 late maximum 而被排除。24 个选中 target 模型在全部合法 prefix 的 eval-valid endpoints 上均通过折叠收据。后者仍不是 query 级、CPU 或量化部署证明；折叠失败的模型仍可作为未合并研究模型参与 local query 比较，但不能据此宣称为可部署折叠模型。

研究判断是暂不替换正式骨架。下一步候选是受限低秩方向残差，并优先进行 output 校准或采用更稳定的 prefix validation；这只是建议，尚未实现或验证。未使用 association-profile 特征，也没有 EvalAI 结果。

可从已验证的 [analysis summary](../results/analysis/unit_session_iteration_20261004/summary.json)、[comparison CSV](../results/analysis/unit_session_iteration_20261004/comparison.csv) 和 [静态图](../results/analysis/unit_session_iteration_20261004/comparison.png) 复查本节数值。重新运行只读分析器的命令为：

```bash
PYTHONNOUSERSITE=1 \
/home/xinyuan/miniconda3/envs/spint/bin/python scripts/analyze_unit_session_iteration.py \
  --root results/unit_session_iteration_20261004 \
  --output results/analysis/unit_session_iteration_20261004 \
  --expected-fit-seal-sha256 d54522362911480a0289c0dd9f87357757cc318a3c5517d58b8de238f26d7c8b
```

## 运行

以下命令从 `SSM/` 目录执行。`PYTHONNOUSERSITE=1` 和 `PYTHONPATH` 固定解释器与隔离的 Mamba 依赖路径；launcher 按配置将 M1/M2 放入各自的 GPU 队列。

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH="$PWD/.tools/mamba_deps:$PWD" \
/home/xinyuan/miniconda3/envs/spint/bin/python scripts/run_unit_session_iteration.py \
  --config configs/unit_session_iteration_20261004.json \
  --output-root results/unit_session_iteration_20261004 \
  --phase fit
```

fit 完成后，从输出记录 `fit_seal_sha256=...`，将该值独立保存。仅在该外部绑定值已确定后执行评分；下面的 `<FIT_SEAL_SHA256>` 是必须替换的字面占位符。

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH="$PWD/.tools/mamba_deps:$PWD" \
/home/xinyuan/miniconda3/envs/spint/bin/python scripts/run_unit_session_iteration.py \
  --config configs/unit_session_iteration_20261004.json \
  --output-root results/unit_session_iteration_20261004 \
  --phase score \
  --expected-fit-seal-sha256 '<FIT_SEAL_SHA256>'
```

上述本地开发结果已通过独立评分审计。它们支持保留该前端作为消融与后续受限适配的研究实现，暂不替换正式骨架。
