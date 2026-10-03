# POSSM / SSM 文献审计（2026-10-03）

本记录只核对现有两份 brainstorming 文档中的可验证主张；“可行候选”是待实验的设计，不应写成已有结论。

## 可追溯的一手来源

- POSSM 论文（NeurIPS 2025）：[arXiv:2506.05320](https://arxiv.org/abs/2506.05320)，[OpenReview](https://openreview.net/forum?id=1i4wNFgHDd)，[项目页](https://possm-brain.github.io/)。截至本审计日，项目页明确标为 **“Code coming soon!”**；论文只称代码将通过 `torch_brain` 发布。官方站当前未提供已发布代码链接，本审计亦未定位到可核验的正式 release；不能将任意复现或 `torch_brain` 的预期发布当作官方实现。
- Gated DeltaNet：[ICLR 2025 论文 / OpenReview](https://openreview.net/forum?id=r8H7xhYPwz)，[arXiv:2412.06464](https://arxiv.org/abs/2412.06464)，[NVIDIA 官方实现](https://github.com/NVlabs/GatedDeltaNet)。
- LinOSS：[ICLR 2025 Oral / arXiv:2410.03943](https://arxiv.org/abs/2410.03943)，[作者官方实现](https://github.com/tk-rusch/linoss)。
- Mamba-3：[ICLR 2026 论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html)，[arXiv:2603.15569](https://arxiv.org/abs/2603.15569)。
- 硬件约束的原始证据：[Zhang et al., *Nature Communications* 2026](https://www.nature.com/articles/s41467-025-68227-w)；量化的原始证据：[Quamba, ICLR 2025](https://arxiv.org/abs/2410.13229)、[Quamba2, ICML 2025](https://arxiv.org/abs/2503.22879)。

## POSSM：实际架构、数据与评测协议

**架构。** 一条 spike token 是 `(UnitEmb(i), t_spike)`：单元 ID 是可学习 embedding，时间用 RoPE。每个通常为 50 ms 的连续 chunk 含可变数目的 spike token。一个可学习 query 对该 chunk 的 K/V 做交叉注意力，产生单一 latent `z_t`；循环主干更新 `h_t=f(z_t,h_{t-1})`。论文试验了 S4D、GRU 和 Mamba 主干。行为读出再以最近 `k=3` 个隐藏态为 K/V，用“RoPE 时间 + session embedding”的 query 做输出交叉注意力，故可在 chunk 内多个时刻、甚至预测未来时刻输出。输入交叉注意力和读出仍是注意力模块；“常数时间”只适用于跨 chunk 的递归状态，而非随 chunk 内 spike 数增长的整条网络成本。

**NHP 数据和切分。** 预训练 `o-POSSM` 用 4 个猴子 reaching 数据集、148 sessions、约 6.70 亿 spikes、26,032 units，覆盖 M1/PMd/S1 与 centre-out、random-target、maze。一个训练样本为 1 s 的 spike 与同段二维手速度，拆成 20 个不重叠的 50 ms chunk；训练窗口不按 trial 对齐。测试在完整 trial 上进行，举例其长度至少为训练序列 3 倍、最高 5 倍。预训练测试含：同猴 C 的新日期、未见猴 T、以及未见数据集 H；指标为跨 session 的平均 `R² ± SD`。论文也报告 20 ms chunk，但这不是 event-by-event 版本。

**适配和训练。** UI（unit identification）冻结所有主干权重，初始化并以梯度训练**新的 unit 和 session embeddings**，通常更新总参数少于 1%。NHP 的 UI 训练总计 500 epochs；FT 则先 UI 100 epochs，再解冻全模型 400 epochs，总计同为 500 epochs。NHP 的单 session 使用 1×RTX8000，跨数据预训练使用 4×H100；单 session 训练少于 30 min，预训练约 36 h。SS/FT 的 input dim 为 64/256、hidden dim 为 256/512、SSM 层数为 1/4；S4D 模型参数量 0.41M/4.56M（GRU 0.47M/7.96M；Mamba 0.68M/8.96M）。

**已报告的实时性和外推边界。** CPU 上 SS 与 `o-POSSM` 分别约 2.44、5.65 ms/chunk；论文称 GPU 最多快于对比 Transformer 9×。这是离线、因果式评测，论文明说当前评估仍是 offline，并没有闭环植入式系统的端到端 latency/energy 测量。人类手写确有跨物种预训练，但语音实验只有归一化 spike counts：不能原样使用精确 spike-time tokenization，读出也改为 strided 1D convolution。因此不能把所有 POSSM 结果概括成“同一 spike-token 架构已在临床在线 BCI 运行”。

## 对现有 brainstorming 主张的更正

| 原主张 | 审计结论 | 依据与正确表述 |
|---|---|---|
| “DeltaNet/Gated DeltaNet 状态即在线回归；数学本质与 APST 闭式岭回归相同。” | **不成立。** | Delta rule 的 `S_t` 对当前 key 的残差做门控的一阶更新（LMS/在线梯度下降视角）；没有维护 RLS 所需的逆协方差 `P_t`，也没有递归 Sherman–Morrison 校正，因而不等于闭式 ridge 或 exact RLS。可称为“可元学习的、近似在线校准记忆”，不能称“同一件事”。|
| “Mamba-2 实数标量 A 表达不了旋转，故不适合运动皮层。” | 前半句只对**单一一阶实标量模态**成立；后半句过强。 | 一个实标量极点不能独立地产生二维旋转；但多 head、输入/输出投影、非线性、层间组合可表示或近似旋转动力学。POSSM 的完整网络也含 attention/readout，不能从 scalar modes 推出整个网络“不能学习旋转”。应以同参数预算的消融实验判断。|
| “POSSM-S4D 最好证明复数/振荡动力学。” | **不是证明。** | 表格只比较特定实现、预算、优化和数据条件；S4D 的优势可来自参数量、初始化、优化或其他结构差异。它只能提出“值得检验”的假设。|
| “没有人在侵入式运动解码 + SSM-CIM + 跨 session 适配，所以是空白。” | **不能由有限检索证明。** | 至少应改为“本次检索尚未发现直接同时覆盖三者的一手论文”。absence-of-evidence 不能作为 novelty 结论；投稿前需系统检索（数据库、引文链、专利）并限定检索日期和关键词。|
| “50 ms chunk 是延迟下限，改 event SSM 即可消除。” | 需要限定。 | 数据积累窗口确实贡献约 50 ms 的观测等待，但 POSSM 的 attention tokenizer 也需重构为事件更新；只替换 SSM 不会自动消除 tokenization/输出调度延迟。|
| “Wei Lu 论文可直接支持 BCI 存内实现。” | 仅支持 co-design 的可行性，不支持 BCI 性能。 | 原论文任务是异步音频/视觉事件流；其模型特意限制为实数、静态对角、每 block 一种或少数固定衰减，并承认这失去 Mamba 的输入选择性。它没有 POSSM、侵入式神经记录或跨 session 适配实验。|

## 三个可辩护的、超出 POSSM 的候选

### 1. 阻尼振子 / LinOSS 主干（优先验证动力学，而非宣称神经科学必然性）

对每个 mode 用实数二阶状态代替一阶标量：

`ṗ = q,   q̇ = −Ω²p − Γq + Bz_t,   y_t = Cp_t`，其中 `Γ ⪰ 0`；以稳定的 IM/IMEX 离散化更新。LinOSS 的一手论文以 forced harmonic oscillators 为基础，证明稳定性（非负对角状态矩阵条件）和因果连续算子的通用逼近性，但这些是其模型性质，**不是**对运动皮层频段或 POSSM 改进的证据。

可在 POSSM 中只替换 S4D 主干，并对齐 hidden/state 参数和训练预算，与实数对角 S4D、原 S4D 比较 `R²`、跨 session calibration curve 与长 trial 外推。硬件代价是每个振子 mode 维护 2 个状态，离散 `2×2` block 需约 4 个 state-MAC（或固定系数时用共享乘法器）；相对一阶实对角递推约翻倍状态 SRAM 和递推算术。若以复数实现，等价为实部/虚部双通道，CIM 需 2×2 real decomposition 或数字复乘，不能宣称“免费”。

### 2. **Exact-RLS** 校准读出（与 DeltaNet 明确区分）

冻结 POSSM tokenizer/SSM，令 `φ_t` 是从其状态读出的低维特征，校准标签为 `y_t`。每 session 使用真正的 ridge/RLS 读出：

`g_t = P_{t-1}φ_t / (λ + φ_tᵀP_{t-1}φ_t)`

`W_t = W_{t-1} + (y_t − W_{t-1}φ_t)g_tᵀ`

`P_t = λ⁻¹(P_{t-1} − g_t φ_tᵀP_{t-1})`，`P_0=α⁻¹I`。

这才是带遗忘的 exact RLS（`λ=1` 对应固定样本的 ridge 递推），应与梯度 UI、APST 的原协议及 Delta-rule memory 分开评估。它的代价也是关键限制：特征维度 `d` 时需保存 `d²` 个 `P` 元素、每个有标签样本约 `O(d² + d×output_dim)` MAC 和一标量倒数；若 `d=512`，仅单精度 `P` 约 1 MiB，显著大于“几个寄存器”。面向植入硬件应先检验 block-diagonal、低秩 `P≈LLᵀ+D` 或 `d≤32–64` 投影，且必须报告其相对 full RLS 的误差。静态投影/主干可在 CIM，`P` 的频繁读写、除法和校准状态更适合 SRAM 近存数字单元。

### 3. 可变步长、二阶离散化的稳定 SSM（Mamba-3 思路的可控子集）

保持 POSSM 的 cross-attention，但令每个 chunk 的真实长度或可靠度产生受限步长 `Δ_t`；对线性主干采用 trapezoidal/Tustin：

`h_t = (I − Δ_t A/2)⁻¹[(I + Δ_t A/2)h_{t−1} + Δ_tBz_t]`。

Mamba-3 的一手论文确实报告“更富表达力的 SSM 离散递推、complex state update、MIMO”，以及同等困惑度时约半 state size 的实验；它没有在 BCI、可变采样或 CIM 上证明收益。因此第一阶段应仅做 `Δ_t` 固定/实测长度/由输入预测三组，并与 ZOH/Euler 对齐参数、在 20/50 ms 及模拟丢包上评估。对实数对角 `A=a_i`，每 state 更新只多出预计算或查表的 `(1−Δ_ta_i/2)⁻¹` 与两次乘加；若 `Δ_t` 每步变化，需每维 LUT/reciprocal 或按少数时间常数共享，且对 CIM 的“固定物理衰减”假设不再直接成立。可把 `Δ_t` 量化为少数档位、把递推留在数字近存单元；MIMO/complex 版本的硬件收益须以完整状态读写和 ADC/DAC 计入，不能从 LLM decode 直接外推到小型 BCI。

## 硬件与量化：可引用的边界

Zhang et al. 的实测 CIM 贡献是：静态 `B̄x` VMM 映射至 RRAM crossbar，WOx short-term-memory 器件的自然衰减实现对角状态项，且只有 layer/block 共享的固定衰减；GELU/sigmoid 等仍为数字 LUT。它的效率来自**输入事件稀疏性**，论文明确说没有强制内部 SSM activation 稀疏；并提示外部控制可在未来调节 time constant。该论文可支持“静态对角 SSM 的异步硬件 co-design 候选”，不能支持选择性 SSM 或 POSSM attention 全部映射到 CIM。

Quamba 证明的是特定 Mamba 系列的 W8A8 PTQ，可在 Orin Nano 的 Mamba-2.8B 上将生成 latency 降低 1.72×、平均 zero-shot accuracy 下降 0.9%；Quamba2 扩展至 W8A8/W4A8/W4A16，并在其模型/平台报告速度和内存结果。这些不是 0.4–9M 参数 POSSM、RLS 状态或模拟 RRAM 的精度保证。任何“Hadamard/旋转后量化可直接复用”的表述应改成待校准集量化误差、长序列状态漂移、ADC/DAC 与 session embedding 写入共同验证的工程假设。

## 建议的证据门槛

先以原 POSSM 50 ms 因果 protocol 重现同一 NHP 切分；只改一个模块，分别报告每 session `R²`、校准 labels/trials、CPU 单步时间、参数/state bytes。候选 1 再报告长 trial stability；候选 2 必须对 full-RLS 与受限 RLS；候选 3 必须对真实/模拟不规则 `Δ`。硬件主张应另报告 batch=1 每事件/每 chunk能耗、state SRAM、校准写次数及 ADC/DAC 开销。通过这些消融后，才可以把“候选”升级为方法贡献或 novelty 主张。
