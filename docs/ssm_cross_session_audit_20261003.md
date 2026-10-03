# SSM cross-session 解码审核与改进路线（2026-10-03）

范围：审核 `ssm_decode` 的 Mamba-3 cross-session 实现和 PEFT round-1 正式产物（50 个主矩阵运行 + 6 个容量对照），参考方向为 POSSM、projection LoRA 和 state-offset。本次只新增 3 个**只评估**的诊断：不训练、不更新任何参数，query 标签只用于最终评分。脚本和输出在 `results/analysis/audit_20261003/`（见 §6）。

## 0. 结论先行

1. **当前最大的误差来源是输入归一化合同，不是 SSM 结构或适配方法。** 对固定的 source-only checkpoint，只把 target 输入改成用 support 段神经数据（无标签）做逐通道 z-score，直接解码 all-valid R² 在 M1 从 0.386 升到 **0.674**，在 M2 从 −0.006 升到 **0.234**。改用严格因果的滑动（EMA）z-score 后，M1 为 **0.699**，M2 为 **0.266**。M1 的结果超过 round-1 全部 9 种适配方法（最好的是 LoRA 0.586）；M2 与最好的 PEFT（state-offset 0.266）持平。两种做法的可训练参数都是 0。
2. **rank-4 projection LoRA 在结构上表达不了这类漂移。** 逐通道 gain 变化是满秩的对角变换，均值漂移需要 bias，而 `LoRALinear` 冻结了 base bias。round-1 的 LoRA、B/C LoRA、state-offset 实际上都在用低秩参数近似一个对角输入校正。
3. **PEFT 方法排名被欠收敛混淆。** lr=1e-4、1000 step 下，48 个适配 fit 中有 31 个的 best_step ≥ 850，18 个恰好落在第 1000 步。full FT 在第 400 步全量解冻后，于第 500–550 步达到峰值，之后 prefix-val 下降。所以表格比较的主要是“同一 lr 下谁收敛得快”。
4. **checkpoint 选择和 support 位置都有偏差。** M1 的 support 只占 session 前 6% 的时间（160,436 个 bin 中的前 10,361 个），M2 占前 14%。M1 的 prefix-val R² 约 0.78–0.81，query 只有 0.53–0.59。M2 上 prefix-val 最高的方法（LoRA）在 query 上并不最好。
5. **source 预训练过拟合，而且缺少 session 结构。** 2.75M 参数的模型只在 3 个（M1）或 6 个 run（M2）上训练。M2 的训练 loss 降到 0.040（归一化 MSE，相当于训练 R² 约 0.96），source-val R² 只有 0.533。所有 source session 共用一个 normalizer 和同一个 `in_proj`，没有 POSSM 式的逐 session 读入或 session embedding。第二轮调试里，S4D-128×2 + IO 适配已经达到 M1 0.601、M2 0.295，高于所有 Mamba-3 PEFT 结果。
6. **M2 的评分几乎全由运动爆发段决定。** 82% 的目标方差集中在每个 trial 的第 10–25 个 bin；legacy tail（bin ≥ 49）只含约 2% 的方差，不适合用来给方法排名。在运动爆发段，预测方差只有真值方差的 50–59%，存在明显的幅度收缩。
7. **两个实现语义问题会影响下一步。** (a) trial-reset 窗口的左侧零填充，在归一化空间里等于“source 平均放电”，不是空输入；FALCON 和部署场景都是连续解码。(b) 当前 state-offset 加在 Mamba-3 的旋转（去旋转状态）坐标系里，换回原始状态坐标就是一个随累计相位转动的偏移；改成流式解码后，这个相位会从 stream 起点一直累积下去。

建议的路线（§4）：先修归一化合同（P0）和 PEFT 训练合同（P1）；再把 projection LoRA 改成“逐通道 affine + LoRA”（P2）；按 POSSM 的做法给预训练加逐 session 读入、连续窗口，并缩小骨架（P3）；最后在上述基础上重新评估 state-offset，包括坐标系消融、预训练时就带 session offset、以及无标签的 offset 估计（P4）。

## 1. 审核对象与口径

- Source checkpoint：`results/debug_round2/latest/{m1,m2}/cross_session_mamba3_official_w256_l4_n32_ctx128_seed0/best.pt`（官方 Mamba-3 SISO，width 256、4 层、state 32、context 128；lr 1e-3，2000 step，best step 分别为 M1 800、M2 1800）。
- Source / target：M1 用 `20120924/26/27` 预测 `20120928`；M2 用 `2020-10-19/20/27` 的 6 个 run 预测 `2020-10-28-Run1`。
- 适配：target 前 33 个长度 ≥ 50 bin 的 trial 作为 support，其中 26 个训练、7 个验证；之后的 trial 作为 query（M1 380 个，M2 232 个）。
- 主指标：物理单位下的 variance-weighted R²，覆盖 query trial 的全部 eval-valid bin（all-valid）。除非另注，本文数字都是**直接解码**（zero）。

## 2. 发现与证据

### F1 输入统计漂移是主导误差

| Checkpoint | Target 输入归一化 | M1 直接 | M1 +ridge | M2 直接 | M2 +ridge |
|---|---|---:|---:|---:|---:|
| source-only | source 合并统计（round-1 合同） | 0.3862 | 0.4830 | −0.0059 | 0.1209 |
| source-only | support 均值替换 | 0.6037 | 0.5952 | 0.1491 | 0.1590 |
| source-only | **support z-score** | **0.6736** | 0.6727 | **0.2338** | 0.2501 |
| source-only | 因果 EMA z-score，半衰期 3,000 bin（60 s） | **0.6992** | 0.7040 | **0.2659** | 0.2738 |
| source-only | 因果 EMA z-score，半衰期 15,000 bin | 0.6992 | 0.7013 | 0.2461 | 0.2605 |
| source-only | 因果 EMA z-score，半衰期 50,000 bin | 0.6908 | 0.6913 | 0.2382 | 0.2540 |
| round-1 最好的 PEFT（3 seed 均值） | source 合并统计 | 0.5861（LoRA） | 0.5824 | 0.2660（state-offset） | 0.2645 |

数据本身也支持这个解释。按 source 标准差计，target 逐通道均值偏移的中位数在 M1 为 0.144、M2 为 0.045，但最大值分别达到 4.1 和 5.5；target/source 标准差比的 5–95% 分位，M1 为 [0.39, 2.03]，M2 为 [0.61, 2.00]。也就是说，少数通道发生了很大的 gain/offset 变化，经过固定的线性 `in_proj` 后，会给所有 latent 注入大误差。

口径边界：

- support z-score 只用 33 个 support trial 的神经数据，不用标签。round-1 的 PEFT 本来就能访问这些数据，所以它完全在原 support 合同之内，和 PEFT 可以直接比较。
- EMA 归一化因果地使用了 query 时段的神经数据（只用严格过去的 bin，不用标签），属于**无监督的测试时适配**协议，必须单独报告。3 个半衰期都是在 query 上看过的，只能当探索性结果；下一轮应预先固定（例如 60 s），或在其他 target 上选。
- 只有一个 source checkpoint（seed 0）和一个 target，还需要多 target 复核（§4 P5）。
- 把 target z-score 事后套到 PEFT checkpoint 上（训练和评估的合同不一致，仅供参考）：M1 LoRA 0.6927、state-offset 0.6889；M2 full 0.2973、state-offset 0.2666。M1 上 PEFT 在重归一化之后只多出约 0.02；M2 多出约 0.06。这提示：先去掉输入漂移之后，留给 SSM 适配的余量比 round-1 表格显示的小得多，在 M1 上尤其如此。

### F2 projection LoRA 的结构限制

- `in_proj` 的 LoRA 增量是 ΔW = A·B（rank 4）。逐通道 gain g 对应的修正是 W·diag(g) − W = W·diag(g−1)，它的秩等于发生变化的通道数，最多可达 C = 64（M1）或 96（M2）。
- 均值漂移需要更新 bias。`LoRALinear` 冻结 `base.bias`，只能通过 ΔW·x̄ 间接产生一个低秩的偏置。
- `io` 训练整个 `in_proj`（含 bias），表达能力足够，但在 lr 1e-4 下 best step 落在 800–1000，没训完。
- core 内的 LoRA（`blocks.i.ssm.in_proj` 整矩阵）同时改动 z、x、B、C、dt、A、trap、angle。round-1 文档已经指出这一点。
- 因此 round-1 回答“该适配 SSM 的哪一部分”时，主导的输入侧漂移还没去掉。表格衡量的主要是：每类低秩参数化逼近对角输入校正的能力有多强。

### F3 PEFT 欠收敛与学习率

| 统计（不含 none） | M1 | M2 |
|---|---:|---:|
| best_step ≥ 850 的 fit | 17 / 24 | 14 / 24 |
| best_step = 1000 的 fit | 9 / 24 | 9 / 24 |
| full FT 的 best_step | 500–550 | 550 |

- state-offset 在 M1 上 3 个 seed 全部停在第 1000 步，prefix-val 曲线是 0.68（250 步）→ 0.75（500 步）→ 0.78（1000 步），仍在上升。
- full FT 在第 400 步全量解冻后很快达到峰值，然后下降（M1 prefix-val 从 0.81 降到 0.77），说明对 26 个 trial 来说全量参数过拟合得很快。
- LoRA 和 offset 这类小参数适配器通常需要比全参微调更高的学习率，这里却和 full 共用 1e-4（full 新解冻的层组还只用 5e-5）。

### F4 checkpoint 选择与 support 位置

- M2 上 prefix-val 最高的是 LoRA（0.373–0.402），query 只有 0.234–0.253；full 的 prefix-val 最低（0.297–0.305），query 反而是 0.249–0.255。7 个验证 trial 选出的 checkpoint 和 query 表现不一致。
- M1 按 query 时间四分位计算 R²（3 seed 均值）：LoRA 为 0.667 / 0.527 / 0.569 / 0.582，source-only 为 0.440 / 0.345 / 0.368 / 0.390。离 support 最近的四分位要高出约 0.1，说明 session 内部存在漂移。EMA 归一化（能追踪漂移）在 M1 上比只用 support 统计高 0.026，与这个解释一致。
- support 只来自 session 开头，所以 prefix 验证得到的分数系统性地偏乐观。

### F5 source 预训练

- 训练日志：M1 训练 loss 从 0.25（第 800 步，best）降到 0.10（第 2000 步），source-val R² 0.785 → 0.774；M2 训练 loss 0.040，source-val 0.533。模型处于明显的过拟合区间。
- M2 的 source 数据合计约 98k 个 bin（约 33 分钟），对应 2.75M 参数。
- 所有 source session 共用一个 normalizer，所以各 session 之间的偏移也留在了训练输入里，骨架被迫去吸收它们。`in_proj` 是共享的，没有逐 session 的读入层或 session embedding，骨架学到的是多个 session 的平均映射。
- 第二轮调试的同口径结果：S4D-128×2（26 万参数）+ IO 适配达到 M1 0.601、M2 0.295（ridge，all-valid）；Mamba-3 256×4 加任何 PEFT 都没超过它。

### F6 评分结构（来自已保存的预测）

M2（5 种方法、3 seed 的典型值）：

| trial 内位置（bin） | bin 占比 | SST 占比 | SSE 占比 | 预测方差 / 真值方差 |
|---|---:|---:|---:|---:|
| [0, 10) | 16% | 12% | 9–10% | 0.47–0.56 |
| [10, 25) | 25% | **82%** | 82–84% | 0.36（source-only）/ 0.50–0.57（适配后） |
| [25, 50) | 41% | 4% | 5–6% | 0.53–0.73 |
| ≥ 50（legacy tail） | 18% | **2%** | 2% | 0.17–0.27 |

M1：[50, 128) 区间占 58% 的 bin、84% 的 SST，适配后预测方差约为真值的 0.51–0.62。

含义：

- M2 的 legacy tail 基本就是运动结束后的静止段，只占约 2% 的方差，R² 接近噪声。它应降级为诊断指标，不再用于方法排名。
- M2 的运动发生在 trial 的第 10–25 个 bin，此时模型只有 10–25 个 bin 的 trial 内历史；trial-reset 窗口把 trial 开始前的准备期神经活动全部截掉了。第二轮调试里，在 M2 上用 recording 连续窗口重新训练 S4D-128×2，结果从 0.094 升到 0.267（单 seed）。
- 预测幅度收缩一半左右。support 上拟合的 ridge 残差校正只带来很小的收益，说明这更可能是信息不足，而不是单纯的输出尺度问题，暂时不应当作可以直接修掉的偏差。

### F7 窗口合同

- 训练和评估都使用右对齐的 128-bin 窗口，在 trial 起点截断，左侧补零。补零发生在归一化空间里，0 等于“source 平均放电”，经过 `in_proj` 的 bias 后，变成一段持续输入。所以 trial 起点时的 SSM 状态，是对 (128 − k) 个 bin 合成平均输入的响应，而不是零状态。训练和评估一致，不构成泄漏；但 target 发生漂移时，这段“平均输入”本身也随之偏移。修好归一化之后，这个问题会减轻。部署时应改用真实的过去上下文，或在填充位置显式屏蔽（例如把 token 置零，并令 dt = 0）。
- FALCON 官方评测在整段 recording 上连续解码，APST 也是如此。trial-reset 合同既偏离部署，也偏离 APST。
- 每个 endpoint 都重新计算一个 128-bin 窗口，没有使用官方的 `step()` 流式 API。round-1 文档已记录这一点。

### F8 state-offset 的坐标系

Mamba-3 的复数状态递推 h_t = α_t R(θ_t) h_{t−1} + B_t x_t、y_t = C_tᵀ h_t，在官方 kernel 里用 RoPE 技巧实现：令 h̃_t = R(Φ_t)ᵀ h_t，其中 Φ_t = Σ_{s≤t} tanh(angle_s)·π·Δ_s（见 `mamba3_siso_combined.py`：`Angles_Cumsum = cumsum(Angles * DT) mod 2π`），然后用 R(Φ_t) 同时旋转 B 和 C。

- 当前实现：`y += silu(z) ⊙ (R(Φ_t)(C_t + b_C))ᵀ h'`。这是在去旋转状态 h̃ 上加常数偏移，换回原始状态坐标就是 R(Φ_t) h'，一个随累计相位转动的偏移。
- 忠实于 State-offset Tuning 原式 y_t = C_tᵀ(h_t + h') 的版本是 `y += silu(z) ⊙ (C_t + b_C)ᵀ h'`，不做旋转。
- 在固定的 128 窗口里，Φ_t 只在 128 个 bin 上累积，范围有界，模型可以学会适应。改成流式解码后，Φ_t 从 stream 起点一直累积，当前版本的 offset 贡献会变成非平稳的。流式场景下，只有不旋转的版本有明确定义。
- 官方配置 `rope_fraction=0.5` 只旋转一半的状态维度，另一半维度上的 offset 本来就是常数。
- round-1 的 state-offset 方法还附带了根层 `in_proj`/`out_proj` 的 rank-4 LoRA，所以它同样表达不了逐通道 gain（F2）。

## 3. 对照 POSSM：哪些可以迁移到固定通道、分 bin 的 FALCON 输入

| POSSM 组件 | 当前实现 | 建议的对应做法 |
|---|---|---|
| unit embedding + spike token + 输入 cross-attention | 固定通道的共享线性 `in_proj` | 逐 session 读入（C→d），或共享 W 加逐 session 通道 affine。分 bin 输入下，unit embedding 等价于 W 的列：z = Σ_i x_i e_i |
| session embedding | 无 | 每个 session 一个向量（作 bias 或 FiLM），target 学一个新向量；它也是 SSM 内部 state-offset 的天然对应物 |
| UI：冻结骨架，重新学习 unit/session embedding，500 epoch | `io`，lr 1e-4，1000 step，未收敛 | 只训读入层和 session 向量，提高 lr、延长训练，直到验证曲线饱和 |
| FT：UI 100 epoch 后全量解冻 400 epoch | `full`：第 200 / 400 步逐步解冻 | 保留，但以 UI 收敛后的模型为起点 |
| 148 个 session 预训练 | 3 个 session（M1）/ 6 个 run（M2） | 规模比不了，应缩小骨架，并用逐 session 读入把每个 session 都用好 |
| 1 s 训练窗口，不按 trial 对齐；测试在完整 trial 上 | 128-bin 窗口，在 trial 起点截断 | 跨 trial 边界随机裁剪 recording 连续窗口；评测改为流式 |
| 输出 cross-attention（以 session embedding 为 query） | LayerNorm + 线性层 | 保留线性读出（硬件友好），可加逐 session 输出 affine |
| D.9：embedding LoRA ≈ UI（0.748 对 0.746） | 根层 `in_proj` 上的 rank-4 LoRA | POSSM 的 LoRA 作用于 unit embedding，相当于逐 unit 的满秩重参数化；我们的版本缺少逐通道自由度（F2） |

关键点：POSSM 的 UI 重新学习的是**每个 unit 的** embedding，本质上就是满秩的逐通道重参数化。在分 bin 输入下，与它最接近的最小版本是“逐通道 affine（+ session 向量）”，而不是低秩 LoRA。

## 4. 改进路线（按优先级）

### P0 归一化合同（成本最低、收益最大）

1. source 训练时，每个 source session 用自己训练 trial 的统计量单独做 z-score，不再合并。
2. target 主协议用 support 段 z-score（只用神经数据，不用标签）；因果 EMA z-score 作为**单独的**无监督协议，半衰期预先固定（例如 3,000 bin = 60 s），不在 query 上选。
3. 在新合同下重跑 none / io / lora / full / state-offset。门槛：如果 source-only 加 target z-score 仍然不低于 PEFT，下一轮 PEFT 要回答的就只剩剩余漂移。

### P1 PEFT 训练合同

1. 按方法族扫学习率，只用 prefix-val 选：LoRA / offset / affine 用 {3e-4, 1e-3, 3e-3}，io 用 {3e-4, 1e-3}，full 用 {1e-4, 3e-4}。step 上限 2000；要求 best_step < 0.8 × 上限，否则延长。
2. 验证集：7 个 trial 的连续块不可靠。改为在 33 个 support trial 上做 k-fold，或交错抽取验证 trial（例如每 5 个取 1 个），两种都报告；也可以在另一个 target 上预先确定固定 step 数。
3. 报告 support 数量曲线（8 / 16 / 33 / 66 个 trial），以及 best_step 的分布。

### P2 重新设计 projection LoRA

1. **逐通道 affine + LoRA**：在 `in_proj` 之前加 x' = g ⊙ x + b（初始化 g = 1、b = 0），参数量为 2C（M1 128，M2 192），再在 `in_proj`/`out_proj` 上加 LoRA（core 的 in/out 可选）。推理时可以合并进 `in_proj`：W' = W·diag(g)，bias' = W·b + bias，没有额外推理开销；在硬件上就是每通道一对 gain/offset 寄存器。
2. 在 LoRA 方法里允许训练 `in_proj` 的 bias（d = 256 个参数）。
3. P0 完成后，扫 rank 1 / 4 / 16，分别只加在 `in_proj` 或只加在 core 上，定位剩余漂移在哪一层。
4. B/C、dt、A、angle 的分行消融推迟到 P0 之后再做；输入侧漂移去掉之前，这些消融的结论不可靠。

### P3 POSSM 式预训练

1. **逐 session 读入**：每个 source session 一个 `in_proj_s`（或共享 W 加逐 session 通道 affine / bias），骨架共享。target 的读入用 source 读入的均值（或最近的 session）初始化，先只训读入层和 session 向量（相当于 UI），再可选逐步全量微调。
2. **连续训练窗口**：在 recording 内跨 trial 边界随机裁剪 128–256 bin 的窗口，burn-in 之后对所有 eval_mask bin 计 loss；评测也用连续上下文，与 FALCON 一致。
3. **缩小骨架、加强正则**：Mamba-3 128×2 或 S4D 64–128，与 256×4 在相同预算下对比；提高 dropout / weight decay，并配合早停。
4. **漂移增强**：训练时随机加逐通道 gain（对数正态，σ ≈ 0.3，对应观测到的 0.4–2.0 标准差比）、逐通道 offset 和通道 dropout，让骨架对这类变化更鲁棒。

### P4 state-offset 方案

1. **坐标系消融**：比较不旋转版本（原始状态坐标，忠实于原论文）和旋转版本（当前实现）。两者都去掉根层 LoRA、都加上 P2 的通道 affine，单独测量 offset 的贡献。
2. **预训练时就带 session offset**：每个 source session 学一个逐层 offset（配合 P3 的逐 session 读入），骨架因此学会使用这个调节轴；target 学一个新 offset，用 source offset 的均值初始化。这比事后加一个零初始化的 offset 更有意义，也就是把 POSSM 的 session embedding 放进了 SSM 内部。
3. **无标签的 offset 估计**：用 target 的无标签神经数据，匹配各层 SSM 输出（`out_proj` 之前）的均值与 source 的均值，闭式求解 h'（最小二乘），并可以像 EMA 那样因果更新。与输入侧的 EMA 归一化对比，检验 offset 在输入校正之外是否还有净收益。
4. **流式前提**：offset 进入连续解码之前，先验证 `step()` 的 angle / ssm / k / v 状态与分块前向一致（round-1 文档已列为待办）。
5. **硬件**：offset 是每层一组寄存器 h'（H·P·N 个，或低秩形式），每步多一次 C̃ᵀh' 点积；不旋转的版本不需要额外的旋转硬件。

### P5 评测

1. **多 target**：在 held-in session 之间轮换 target（M1 有 4 个 session，M2 有 4 天 / 7 个 run），其余 session 作 source；同时报告“只用更早 session”的时间前向版本。按 target 报告配对差值。
2. **指标**：all-valid 为主指标（与 FALCON 接近）；M2 的 legacy 降为诊断。把按 trial 内位置、按 query 时间四分位的分解作为标准输出（直接用 `cohort_structure.py`）。窗口改为连续之后，再加整段 recording 的 eval_mask 指标。
3. APST 的 FALCON held-out 官方成绩（M1 0.654、M2 0.423）只作背景参考：协议、session 和数据划分都不同，不能直接比较。

## 5. 建议的下一轮最小实验矩阵

| 阶段 | 内容 | 规模（估计） | 判定门槛 |
|---|---|---|---|
| A | 固定现有 source checkpoint，target 用 support z-score；方法为 none / io / lora / affine / affine+lora / offset（旋转）/ offset（不旋转）/ full；按 P1 扫 lr；3 seed × 2 个任务 | 约 140 个短 fit，每个约 45 s，两张 3090 约 1 小时 | 有方法在 support z-score 的 source-only 基础上带来稳定收益（3 个 seed 方向一致） |
| B | 新 source：逐 session 归一化 + 逐 session 读入 + 连续窗口 + 漂移增强；width 128×2 和 256×4；再跑阶段 A 中最好的 4 种方法 | 4 个 source fit + 约 50 个适配 fit | 新 source 在 source-only 和适配后都优于旧 source |
| C | 前 3 名配置做多 target 轮换和 support 数量曲线 | 视 target 数量而定 | 按 target 配对比较，结论在多数 target 上成立 |

## 6. 本次审核产物与复现

| 文件 | 内容 |
|---|---|
| `results/analysis/audit_20261003/renorm_eval.py` / `.json` | source-only 和 4 种 PEFT checkpoint（seed 0），在 source / support 均值 / support z-score 三种输入归一化下的 all-valid 与 legacy R² |
| `results/analysis/audit_20261003/ema_eval.py` / `.json` | source-only 在 3 种半衰期因果 EMA z-score 下的结果 |
| `results/analysis/audit_20261003/cohort_structure.py` / `.json` | 已保存预测按 query 时间四分位和 trial 内位置的分解：R²、SSE / SST 占比、预测方差与真值方差之比 |

```bash
cd /home/xinyuan/Work_host/SSM
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
ENV="PYTHONNOUSERSITE=1 PYTHONPATH=$PWD/.tools/mamba_deps:$PWD TRITON_CACHE_DIR=/tmp/ssm_audit/.triton CUDA_VISIBLE_DEVICES=0"
env $ENV $PY results/analysis/audit_20261003/renorm_eval.py
env $ENV $PY results/analysis/audit_20261003/ema_eval.py
PYTHONNOUSERSITE=1 $PY results/analysis/audit_20261003/cohort_structure.py   # CPU，只读 predictions.npz
```

局限：只有一个 source seed 和一个 target；EMA 半衰期在 query 上看过；把重归一化事后套到 PEFT checkpoint 上存在训练/评估合同不一致。这些结果都只是方向性证据，需要按 §5 在新合同下重跑确认。
