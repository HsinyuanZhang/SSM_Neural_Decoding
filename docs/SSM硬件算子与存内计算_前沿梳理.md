# SSM 的硬件算子与存内计算：前沿方向梳理（面向 BCI）

> 整理日期：2026-10-01。各工作的数字取自论文摘要或项目页（下文链接），**我没有逐篇读全文**，引用前请核对原文。
> 标注：**【推断】** 表示是我自己的判断，不是文献结论；**【记忆】** 表示来自我的已有知识，这次没有重新检索核实。
> 配套阅读：[POSSM_ASIC_cross-session_分析.md](POSSM_ASIC_cross-session_分析.md)

---

## 0. 一段话结论

目前的工作大致分成五条线：
1. **数字加速器（FPGA/ASIC）**：面向 Mamba 类 LLM，核心手段是量化、非线性近似、稀疏化、扫描融合和投影低秩化。
2. **近存/存内处理（PIM）**：面向数据中心推理。它们的共同发现是**状态更新是访存受限的逐元素运算**。
3. **模拟存内计算（RRAM/忆阻器 CIM）**：投影层放进交叉阵列，**状态衰减直接用器件的物理动力学来实现**。这是最"前沿"、也最接近我们需求的一条线，代表工作是 Michigan Wei Lu 组 2026 年发表在 Nature Communications 上的论文。
4. **神经形态实现**：Loihi 2 上跑 S4D、Spiking SSM，以及 ABR 的 TSP1 商用芯片。
5. **算法向硬件让步**：比如 Mamba-3、实数对角 SSM、固定共享衰减常数。

对 BCI 来说，第 3、4 条线最相关，因为我们是 batch=1、逐 token 的流式推理，而且模型很小。第 1、2 条线大多是在优化 LLM 的 prefill 吞吐，可以借鉴的主要是量化和非线性近似【推断】。

---

## 1. 先拆算子：SSM 推理里到底有哪几类运算

| 算子类别 | 例子 | 计算特征 | 适合的硬件 |
|---|---|---|---|
| **投影（MVM）** | Mamba 的 in_proj/out_proj、S4D 的混合线性层、GRU 的 W·x 和 U·h | 参数和 FLOPs 的主体，权重是静态的 | **最适合 CIM**（交叉阵列天然就是在做 MVM） |
| **状态递推** | `h_t = Ā ⊙ h_{t-1} + B̄ x_t`（对角） | 逐元素运算，算术强度极低（Mamba 解码约 2.5 ops/byte，见 Mamba-3 的报道），**访存受限** | 近存数字逻辑，或者**用器件的物理动力学来实现** |
| **非线性 / 输入相关参数** | Mamba 的 Δ = softplus(·)、exp(ΔA)、SiLU、门控 σ/tanh | 查表或分段线性近似 | 数字逻辑 |
| **扫描（scan）** | 训练和 prefill 时的并行前缀和 | 只在长序列并行处理时才需要 | BCI 的流式推理基本用不上 |

**两种工作模式**：LLM 加速器主要在优化 *prefill*（并行扫描、吞吐），以及大 batch 的 *decode*。BCI 是 **batch=1、每 20–50 ms 一步的 decode**，瓶颈在**每一步的能耗和延迟**，以及**权重存储**。所以判断一篇硬件论文对我们有没有用，先看它优化的是哪种模式。

---

## 2. 方向一：数字加速器（FPGA / ASIC）

| 工作 | 单位 / 会议 | 平台 | 关键技术 | 结果（取自摘要） |
|---|---|---|---|---|
| **MARCA** | 上交 戴国浩组，ICCAD 2024【记忆】 | ASIC 仿真 | 可重构 PE 阵列，同时支持矩阵乘和逐元素运算；可复用的非线性函数单元；缓冲区管理 | — |
| **MARCA-v2** | 上交，IEEE TCAD 2025 | ASIC 仿真 | ΔAh 和 ΔBx 两条路径的**互补静态稀疏**（δ-bitmap）；exp 用"fast biased exponential"，SiLU 用分段线性，复用 PE 来算 | 相对 GPU：prefill 能效提升 8.3–33.5×，decode 能效提升 3.1–27×；精度下降 ≤3.59% |
| **LightMamba** | 北大 李萌组，DATE 2025 | VCK190 / U280 | **旋转辅助量化**（处理离群值）；**SSM 部分用 2 的幂次（PoT）量化**；大部分计算压到 4 bit；计算重排和细粒度 tiling 融合 | 能效 4.65–6.06× GPU；[开源](https://github.com/PKU-SEC-Lab/LightMamba) |
| **FastMamba** | 2025 | VC709 | 线性层先做 **Hadamard 变换**再量化到 8 bit；SSM 部分细粒度 PoT 量化；非线性用一阶线性近似 | Mamba2-130M prefill 比 RTX3090 快 8.9× |
| **eMamba** | Kim…Ogras, Park 等，ESWEEK (CODES+ISSS) 2025 | ZCU102 + **GF 22 nm ASIC** | 用轻量替代件换掉 LayerNorm；近似 SiLU/exp；**近似感知的 NAS**；全流程量化 | 功耗降 9.84×，能耗降 48.6×，面积小 4.77×（对比基线方法）；**面向边缘、小模型**，和我们最接近。UW–Madison 新闻稿提到在 GF 22 nm 上做了"第一颗 Mamba chiplet"（页面没能打开，没有核实细节） |
| **LowRank-SSM** | 2026.08 arXiv | Versal VC1902 | 对投影做截断 SVD，把**秩作为硬件设计变量**；双路径投影 + 融合扫描 | 投影权重存储减少 20%；精度有损（审稿意见指出这一点） |
| **XAMBA** | 2025 | Intel Core Ultra NPU（商用 NPU） | 把 CumSum/ReduceSum 改写成矩阵运算（CumBA/ReduBA）；激活函数用分段线性近似（ActiBA） | 最高 4.8× 加速；[开源](https://github.com/arghadippurdue/XAMBA) |

**算法侧的量化基础**：Quamba（ICLR 2025）和 Quamba2（ICML 2025）是选择性 SSM 的后训练量化方案（8 bit / W4A8 等），上面多数加速器都借鉴了它们的离群值处理思路。

**共同规律**：
- 线性层：先旋转或 Hadamard 变换去掉离群值，再做 4/8 bit 量化。
- SSM 递推：用 PoT 或细粒度量化，把乘法变成移位。
- exp、softplus、SiLU：用分段线性或查表近似，复用现有运算单元。
- 投影：做低秩分解或稀疏化，减少权重存储。

对 BCI 的启发【推断】：这些手段可以直接迁移到 POSSM-SS 规模的模型上。但它们的评测场景是 130 M–2.8 B 参数的 LLM，在 0.5 M 参数的模型上量化误差的行为可能不一样，需要重新验证。eMamba 的"近似感知 NAS + 22 nm ASIC"路线最接近小模型、低功耗的场景。

---

## 3. 方向二：存内处理（PIM，主要面向数据中心）

- **Pimba**（MICRO 2025）的核心发现是：**状态更新不像注意力那样能廉价地按 bank 做 PIM**。它的做法是让每两个 bank 共享一个状态更新单元（SPU），用 MX 格式的量化乘加器，最终 token 吞吐比 GPU 高 4.1×。
- 对我们的意义【推断】：这条线证明了"状态更新应该放在存储器旁边做"。只是在植入端尺度上，对应的是**用小 SRAM 加近存逐元素单元**，用不到 DRAM-PIM。

---

## 4. 方向三：模拟存内计算（RRAM / 忆阻器）——最值得关注

### 4.1 代表工作

**① Zhang, …, Wei D. Lu（密歇根大学），*Compute-in-memory implementation of state space models for event sequence processing*，Nature Communications，2026.01**
- 硬件：4 个 64×64 的 1T1R RRAM 交叉阵列，集成在 65 nm CMOS 上，8 bit DAC/ADC。
- **关键创新：状态节点用 WOₓ 忆阻器的短时记忆效应来做**。器件电导会自发衰减，正好对应 `h ← λh` 这一步。衰减率通过退火时间来调。
- 为了适配器件，模型做了改造：**实数对角 SSM，每个 block 共享一个固定衰减常数 λ**，全网只有 6 种衰减曲线，便于制造。去掉了卷积和复数运算。
- 计算分工：B̄x 这类投影在 RRAM 阵列里算，衰减由器件物理完成，GELU/sigmoid 用数字查表，残差用数字加法器。
- **事件驱动、异步**：只在事件到达时计算，两个事件之间靠器件自己衰减。
- 结果：SSC 84.7%（8 bit 84.4%，加入器件波动后 82.0%），DVS Gesture 97.3%。SSC 估算功耗 34 mW，其中 ADC 占 43.7%。
- 论文没有涉及 BCI。

**② Siegel, Yang, Strachan（Jülich PGI-14），*IMSSA: Deploying modern state-space models on memristive in-memory compute hardware*，arXiv 2412.20215**
- 第一次在**真实忆阻器交叉阵列**（3 个 64×64）上跑 S4D 核。状态通过"阵列输出 → 下一时间步再输入"的时延反馈实现。
- 用量化感知训练把 A 矩阵压到约三值。
- 任务是 SHD 子集上的二分类，硬件 81.69%，软件 95.06%。**模拟非理想性带来的精度差距很明显**。

**③ HPD：Hybrid Projection Decomposition（arXiv 2508.11935，2025）**
- 研究 Mamba 在模拟 CIM 噪声下的鲁棒性。做法是对输出投影做 SVD，把 UΣ 留在模拟阵列上，Vᵀ 放到数字端。
- 启示：**哪些层对噪声敏感、应该留在数字端**，是 SSM 上 CIM 的核心设计问题。

### 4.2 这条线的设计范式【推断，基于以上工作归纳】

```
静态投影（MVM） ──► RRAM 交叉阵列（非易失、零权重搬运）
状态衰减 λ·h    ──► 器件物理（易失忆阻器 / 电容 / 漏电积分）或近存数字
非线性、门控    ──► 数字查表
对噪声敏感的层  ──► 数字（HPD 思路）
```

这其实是把"连续时间 SSM `ḣ = Ah + Bu`"直接映射成**物理系统**。对角实数 A 就是一组 RC 漏电积分器，这也是 SSM 和神经形态 LIF 神经元能互相转换的原因。

---

## 5. 方向四：神经形态实现

- **Loihi 2 上的 S4D**（Meyer 等，arXiv 2409.15022，后发表在 IEEE）：第一个在神经形态硬件上实现的 SSM。在**逐 token** 推理时，和 Jetson Orin Nano 相比能耗低约 1000×，延迟低约 75×。**但在离线批处理时 Jetson 更好**。这个对比和 BCI 的流式场景完全吻合。
- **Spiking SSM**：SpikingSSM（AAAI 2025）、SPikE-SSM、P-SpikeSSM、SiLIF（用 SSM 的参数化来设计 LIF 神经元），以及 *A Second-Order SpikingSSM for Wearables*（2025）。这一类工作把 SSM 的状态和脉冲神经元打通，输出稀疏的二值激活，适合事件驱动的硬件。
- **商用芯片 ABR TSP1**（Applied Brain Research）：专门面向状态空间网络（LMU 系），8 bit 下最多 1000 万参数，ASR <35 mW，应用列表里写了 biosignal classification。这说明 **SSM 专用边缘芯片已经开始商业化**。
- 算法侧【记忆】：LRU（Orvieto 等，ICML 2023）、minGRU/minLSTM（"Were RNNs All We Needed?"，2024）这类简化的线性递归，把门控里依赖 h 的稠密乘法去掉了，在硬件上更便宜。

---

## 6. 方向五：算法向硬件让步

- **Mamba-3**（ICLR 2026，CMU/Princeton/Together/Cartesia）：用 **MIMO** 结构把解码时的状态更新从外积变成矩阵-矩阵乘，提高算术强度；**状态尺寸减半**；改用梯形离散化；通过数据相关的 RoPE 实现复数状态。它的出发点是 GPU 利用率，但"同样性能下状态更小"对片上 SRAM 同样有利。
- **Wei Lu 组的做法**：为了适配器件，主动把模型限制成实数、对角、block 共享的固定 λ，再靠其余可训练参数补偿。这是"让模型迁就器件"的典型例子。

---

## 7. 王中锐组（SUSTech）的相关积累——和你们最近的合作点

| 工作 | 期刊 | 和 SSM / BCI 的关系 |
|---|---|---|
| *Resistive memory-based zero-shot liquid state machine for multimodal event data learning*（一作 Ning Lin，40 nm 芯片） | Nature Computational Science 2025 | **固定随机**的 LSM（脉冲储备池）编码器 + 可训练的投影；任务里**包含"脑机接口的零样本迁移"和"神经与视觉数据对齐"**；训练成本降低 152–393× |
| *Continuous-Time Digital Twin with Analogue Memristive Neural ODE Solver*（作者含 Jichang Yang） | Science Advances（课题组主页标注为 in press） | 在忆阻器上用模拟方式求解连续时间动力学，和连续时间 SSM 是同一类问题 |
| *Echo state graph neural networks with analogue random resistive memory arrays* | Nature Machine Intelligence 2023 | 随机 RRAM 做循环储备池 |
| *Pruning random resistive memory for optimizing analog AI* | Nature Communications 2025/26 | 随机 RRAM 的剪枝和优化 |

**它们的共同思路**：主干固定（随机或预训练好）放在 RRAM 里永远不重写，只训练或适配很小的一部分。这和 POSSM 的"冻结主干 + 只换 unit embedding"、以及 APST 的"冻结全部权重 + 闭式 profile"**在结构上是同构的**【推断】。Jichang Yang 同时出现在 APST 和这篇 neural ODE 论文的作者里，组内已经有可以直接打通的基础。

---

## 8. 对你们方向的启示和可能的切入点【推断】

1. **目前没有人在侵入式运动解码上做 SSM 的存内计算，更没有人把它和跨 session 适配结合起来。** Wei Lu 做的是事件相机和音频，Loihi 上测的是 sMNIST/sCIFAR，王组的 LSM 只涉及部分 BMI 任务，而且用的是储备池，不是 SSM。这个组合是一个空白。
2. **具体架构**：
   - POSSM 的 S4D 主干改成实数对角、按 block 共享 λ（借鉴 Wei Lu 的约束）。衰减可以用易失忆阻器或数字实现。
   - 投影层和冻结主干放进 RRAM 交叉阵列。
   - 每个 session 要改写的 unit 表、K/V 表和打分表放在 SRAM，由 APST 式的闭式校准生成。
   - 非线性、softmax 和 exp 用数字查表。
3. **"一次校准，同时补偿两种扰动"**：模拟 CIM 的器件漂移和电导弛豫，与神经记录的跨 session 漂移，本质上都是"冻结主干遇到了分布扰动"。可以试着让每次 session 校准时的闭式 profile、读出层顺带吸收芯片自身的非理想性。这一点需要实验验证，但如果成立，会是一个有意思的卖点。
4. **时间常数是否匹配**：BCI 用的是 20–50 ms 的 bin，运动相关的动力学大约在 100 ms 到 1 s 量级。需要确认 WOₓ 这类器件的衰减时间常数能不能调到这个范围（Wei Lu 的论文里有调节方法，具体数值要看原文）。
5. **评测**：建议在 FALCON 上同时报告 R²、每步能耗、权重存储，以及每次 session 适配的写入量和计算量。可以对比的硬件基线有：纯数字实现、Loihi 式神经形态实现、RRAM-CIM 实现。

---

## Sources
- Zhang…Lu, Nature Communications 2026: https://www.nature.com/articles/s41467-025-68227-w
- IMSSA (Siegel, Yang, Strachan): https://arxiv.org/abs/2412.20215
- HPD: https://pith.science/paper/2508.11935
- MARCA-v2 (SJTU, TCAD 2025): https://dai.sjtu.edu.cn/my_file/pdf/0a61722b-778e-459c-a5fe-5b01ea5e3661.pdf
- MARCA: https://www.researchgate.net/publication/390634832_MARCA_Mamba_Accelerator_with_Reconfigurable_Architecture
- LightMamba: https://arxiv.org/abs/2502.15260
- FastMamba: https://arxiv.org/abs/2505.18975
- eMamba: https://arxiv.org/abs/2508.10370
- First Mamba chiplet in GF 22nm (UW–Madison): https://chips.wisc.edu/2025/06/12/first-mamba-chiplet-in-gf-22nm/
- LowRank-SSM: https://pith.science/paper/2608.02954
- XAMBA: https://arxiv.org/abs/2502.06924
- Pimba (MICRO 2025): https://arxiv.org/abs/2507.10178
- Quamba / Quamba2: https://arxiv.org/abs/2410.13229 , https://arxiv.org/abs/2503.22879
- Loihi 2 S4D: https://arxiv.org/abs/2409.15022
- Spiking SSM: https://ojs.aaai.org/index.php/AAAI/article/view/34245/36400 , https://arxiv.org/abs/2506.06374 , https://arxiv.org/abs/2410.17268 , https://arxiv.org/abs/2510.14386
- ABR TSP1: https://open-neuromorphic.org/neuromorphic-computing/hardware/tsp1-time-series-processor-applied-brain-research/
- Mamba-3 报道: https://www.marktechpost.com/2026/03/18/meet-mamba-3-a-new-state-space-model-frontier-with-2x-smaller-states-and-enhanced-mimo-decoding-hardware-efficiency/
- 王中锐组主页: https://zhongruiwang.github.io/publication.html
- RRAM LSM (Nature Comp Sci 2025): https://www.nature.com/articles/s43588-024-00751-z
