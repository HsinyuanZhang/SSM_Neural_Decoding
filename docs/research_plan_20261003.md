# 跨 session iBCI SSM：两周研究计划（2026-10-03）

## 决策

推荐主假设：**共享的、量化友好的 real-`2×2` 阻尼振子 codebook 提供跨 session 的低维动力学；对 recording set 置换等变的前端把未见 channel/unit 汇聚为该动力学的输入；只在有行为标签的 calibration 段用 exact RLS 更新小读出，并让同一 profile 同时吸收神经记录漂移与芯片静态误差。**

两句 pitch：现有快速 FALCON pilot 可先在原始 channel 的线性前端上测试“动力学 + 校准”是否值得继续，而完整方案把不固定的记录单元当作一个集合，以共享状态模型而不是固定 channel 身份进行跨 session 解码。若 calibration profile 在独立、带标签的支持段上估计后，同时降低留出 query 段的 neural drift 与注入 chip error 的损失，便得到一条可测而非宣称既有的联合补偿路径。

最强反对意见：跨天神经漂移与模拟器件误差的统计结构未必相同，且 exact RLS 的 `O(d²)` 校准状态可能吞掉小模型的硬件优势；置换等变前端也可能在真实单元身份本身有预测力时损失准确度。计划必须先用严格的 support/query、漂移分离和 state-memory 审计反驳或接受这一反对意见，不能从“二者都是分布偏移”推出可共同补偿。

相关文献边界见 [文献审计](literature_audit_20261003.md)：POSSM 是 spike-token cross-attention + recurrent backbone，官方项目页仍写“Code coming soon”；LinOSS、Mamba-3、DeltaNet 和 CIM 论文可提供模块或硬件约束，均未评估本方案。

## 候选池（12 项）

| # | 候选 | 要检验的命题 | 主要代价/风险 | 去留 |
|---|---|---|---|---|
| 1 | grouped real-`2×2` damped oscillators | 少量共享旋转/衰减模式可比实对角主干更稳地外推 | 每 mode 两个状态和 `2×2` 更新 | 保留 |
| 2 | learned oscillator codebook（每组从有限 `Ω, Γ` 档选择） | 共享时间常数可减少 session 过拟合并便于硬件实现 | codebook 太粗会欠拟合 | 保留 |
| 3 | permutation-equivariant set frontend | 未见/重排的 channel 集合可映射到共同 latent | 丢失稳定 unit identity 的信号 | 保留 |
| 4 | profile-folded set modulation | 小的 session profile 调制 set pooling/输入投影可抵抗 recording shift | profile 泄漏或过拟合 support | 保留，限标签支持段 |
| 5 | label-support-only exact RLS readout | 小维 state feature 的闭式递推可替代长梯度 UI | `P∈R^{d×d}` 读写、标签质量 | 保留 |
| 6 | neural + chip-error joint calibration | 一份 profile 能在两类扰动共存时改善 query | 两类误差不共线，可能相互干扰 | 保留为假设，非既有结果 |
| 7 | Delta-rule calibration memory | 低存储的 LMS 更新接近 RLS | 非 exact RLS、步长敏感 | 对照，不作为主张 |
| 8 | input-selective `Δ_t` | 非规则 bin/丢包时自适应时间尺度 | 固定物理衰减 CIM 不再直接适配 | 次级消融 |
| 9 | Mamba-3 MIMO + complex update | 更小 state 仍保留序列能力 | LLM 结果不能外推，算子复杂 | 暂缓 |
|10| real diagonal shared decay | 最容易映射短时忆阻器 | 压缩旋转动力学 | 硬件下界对照 |
|11| fixed S4D/complex diagonal | 复数线性模态是否已足够 | 复乘和 state 加倍 | 动力学对照 |
|12| GRU/minGRU | 门控循环是否以更少硬件成本胜出 | 不能自然提供 closed-form calibration | 基线 |

## 从候选池筛到五项

1. **主模型：grouped oscillator codebook + set frontend + small exact-RLS readout。** 模式 `j` 的连续形式为 `ṗ_j=q_j`、`q̇_j=−ω_j²p_j−γ_jq_j+B_jz_t`，`γ_j≥0`；离散化必须显式固定并记录。set 前端需满足输入 unit/channel 的重排只重排中间 token、不改变聚合输出；profile 只能由 support 产生。
2. **对照 A：同一 set frontend + real diagonal shared-decay SSM。** 它给出最接近固定衰减、低复杂度 CIM 的下界。
3. **对照 B：同一前端 + complex/S4D 或非共享 `2×2` oscillator。** 用于区分“二阶旋转”收益与“分组/码本约束”收益。
4. **对照 C：同一 frozen backbone + Delta/LMS calibration。** 它检验 exact RLS 的 `P_t` 状态是否物有所值，且禁止把两者混称。
5. **对照 D：现有 raw-channel linear frontend FALCON pilot。** 它是快速局部可行性检查；不具备 set-equivariance、未见 unit 处理或完整 POSSM tokenizer 的结论外推资格。

不在两周主线内的 #8–#12 不是否定其价值，而是因为会混入不规则采样、MIMO、选择性状态、复杂门控或完全不同的硬件假设，妨碍确定最小因果贡献。

## 两条实现/结论轨道必须分开

| 轨道 | 当前/目标输入 | 能回答的问题 | 明确不能声称 |
|---|---|---|---|
| 快速 FALCON pilot | 固定的 raw channel 向量 + 线性 frontend | 在局部同分布或预先约定 session split 下，oscillator/RLS 数值是否稳定、是否值得投入 | 跨 recording-set 泛化、POSSM 等价性、channel 置换鲁棒性、CIM 实测收益 |
| 完整 cross-session 架构 | 可变大小的 `(unit/channel features, timestamp)` set，置换等变聚合 + profile | 未见/变化记录集合上的 support→query 泛化，身份/集合归纳偏置是否有效 | 真实芯片联合补偿或临床在线效能，除非另有硬件/闭环实验 |

## 三个决定性测试

1. **支持集隔离的跨 session 测试。** 每个 held-out session 分为按时间先后的、带标签 support 与完全不参与任何 profile/RLS/normalization 拟合的 query；比较 frozen、gradient UI、Delta/LMS、full RLS。报告每 session `R²`、置信区间、labels 数曲线与失败 session，而非只报均值。
2. **集合归纳偏置测试。** 对 query 中 channel 顺序作置换（输出必须不变），再作 controlled channel dropout、重采样和未见-unit split。比较 raw linear 与 set frontend；若原始 unit ID 可用，另做 identity-ablation，防止“等变”其实只是放弃有用身份信息。
3. **联合扰动的可证伪测试。** 在完全冻结的主干/静态投影上，以预注册的独立形式注入可测 chip-like gain/offset/noise/drift，并分别评估 neural-only、chip-only、both。profile/RLS 只在 support 拟合；要求 `both` 的改进不以任一单扰动显著退化为代价。若 joint profile 不优于两个分开的轻量适配器，就撤销“联合补偿”主张。

每个测试还要记录：RLS feature width `d`、`P` 的 bytes、每 label 近似 MAC、是否发生数值不稳定、推理/校准时间。先以一个固定 seed 的 smoke run 检查数据泄漏和 shape，再做预先规定的多 session/seed 评价。

## 两周 pilot（无“已评估”暗示）

| 时间 | 交付物与停止条件 |
|---|---|
| Day 1–2 | 定义 support/query contract、normalization fitting scope、FALCON 固定 split；实现或核查 real diagonal、`2×2` oscillator 和稳定 RLS 的最小数值单测。若 feature/RLS 维度不能压至可审计内存，先转低维投影。 |
| Day 3–4 | 在 raw-channel linear frontend 上完成单 session 和一个跨日 smoke pilot；输出每项 `P` bytes、condition number/递推失败率和原始结果表。该阶段只决定数值可行性。 |
| Day 5–6 | 加入 set frontend 的 permutation test、channel dropout test 和 identity-ablation；用同一训练预算比较 diagonal 与 grouped `2×2`。若置换不变性测试不通过，停止性能解释。 |
| Day 7–8 | 完成 support labels 扫描，比较 no-adapt / gradient UI / LMS / exact RLS；固定 `λ, α, d` 或把调参严格限制在开发 sessions。 |
| Day 9–10 | 注入并分离 chip-like 扰动；比较 joint profile 与两个独立轻量 profile。若 joint 没有一致收益，结果写作 negative finding，不再主推联合补偿。 |
| Day 11–12 | 多 session/seed 重跑最小胜出组，制作 session-level paired 图、state-memory/MAC 表、失败案例。 |
| Day 13–14 | 结论门：只有在测试 1–3 均通过且 raw/set 结论未混淆时，才推进完整论文式实验；否则保留最快稳健的局部模型，重新选择 #8 或 #12。 |

## 成功判据与预先承诺的解释

- 主模型至少在预先指定的 held-out session 上相对 diagonal 对照取得一致的 paired 改善，同时没有超过确定的校准内存预算；改善仅出现在训练 session 不算成功。
- RLS 必须在相同 support labels 下胜过或匹配 LMS/UI，且其 `d²` 存储与标量倒数成本被量化；若不满足，使用 LMS 只能被称为近似在线更新。
- “联合补偿”只有在 `both` 扰动下相对 neural-only profile 与 chip-only profile 均有可复现优势才成立；只对单一注入噪声有效不支持该主张。
- 若 oscillator 无优势，不能据此断言运动皮层不存在旋转；它只否定此参数化、预算和数据协议下的收益。若 raw frontend 有优势而 set frontend 没有，论文问题应收缩为固定通道 FALCON adaptation，不能包装为 generalizable cross-session decoder。
