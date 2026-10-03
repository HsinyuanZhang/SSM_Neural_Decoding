# Cross-session SSM 迭代结果（2026-10-03）

本轮依据 [审核与改进路线](ssm_cross_session_audit_20261003.md) 完成 A、B 正式实验及官方校准预算的部署探索。A 为 116 个 fit、44 次选中模型评分；B 为 4 个 source fit、78 个适配条目、78 次评分。正式产物均通过用户指定的 `gpt-5.6-sol`、`xhigh` 独立审核。README 保持简短；本轮未增加 SystemVerilog 工作。

## 结论及范围

输入归一化修正后，M1 的旧 source-only 已达到 R² 0.6736，旧 source 上全部适配方法均未超过它。新预训练的小模型达到 0.7229，大模型达到 0.7332；M1 的主要问题不能归因于规模不足。M2 对预训练及容量更敏感：小模型最好为 0.3988，大模型最好为 0.4912。M2 增加容量有价值，前提是输入与连续上下文合同正确。

这些均为一个 held-in target 的本地 cross-session 开发结果，主协议使用 33 个可用 pilot 片段，不能当作 hidden held-out 成绩。M1 官方预算只有 10 个原始 trial；重新执行后，小模型 source-only 为 0.6881，小模型 IO 的三个 seed 明显不稳定，改变了提交候选的选择。

本轮验证了 Mamba-3 SISO 的 session 前端、可折叠 affine、projection LoRA、逐步解冻 full tuning 和两种坐标系的 state-offset，提供了可复核的调试及部署骨架；尚未证明超过 POSSM，也未完成多 target 泛化验证。

## 研究合同

A 固定旧 Mamba-3 SISO source（width256、4层、state32、context128）。M1 source 为 20120924/26/27，target 为 20120928；M2 source 为 2020-10-19/20/27 的六个 run，target 为 2020-10-28-Run1。support 为前33个长度至少50bin的可用片段，26训练、7交错验证。M1 的 neural-only 前导片段被 pilot loader 计入此合同；M2 短 trial 被过滤。因此这里的33不能解释为33个官方原始 trial，部署需另跑合法预算探针。

support 无标签神经数据提供逐通道 z-score，输出统计固定为 source 统计。拟合数据中的 query 标签置为 NaN。主指标是原始物理单位的 variance-weighted R²，覆盖 query 全部 eval-valid bin；ridge 和 legacy tail 仅作诊断。A/B 全部方法具有相同 query 索引及原始标签。

LR 按三个 seed 的 prefix-val 均值选择：IO 为 3e-4/1e-3，LoRA/affine/offset 为 3e-4/1e-3/3e-3，full 为 1e-4/3e-4。从2,000步开始，best_step ≥0.8×预算则延长到4,000/8,000步；到上限仍不满足的候选不能参与选择。一个 LR 必须三个 seed 都通过收敛门槛。none 只有一个确定性条目。A 的三个 M2 晚峰值候选被排除，所选 LR 可从 prefix 产物独立重建。

B 使用 recording 连续因果128-bin窗口，保留真实跨 trial 上下文，每个 endpoint 重新计算固定窗口。没有验证官方 `step()` 或完整缓存状态与窗口前向等价，不应描述为无限历史的流式 SSM。

新 source 同时加入：逐session train-only输入统计、共享投影加session gain/bias/latent embedding、跨trial随机窗口、session-balanced sampling、gain/offset/channel-drop漂移增强、dropout0.2、weight decay0.001。source trainer 未加载 target；导出时把 source 前端均值折叠到普通投影。共享权重列缩放不等同于 POSSM 的完整 unit embedding 重学习，因此 full IO 仍为必要对照。

B 比较旧 source 的连续窗口评估、小模型128×2、大模型256×4；LR从A prefix选择转移。新 source 每个任务/容量只有一个 seed，适配各运行三个 seed。宽度与深度共同变化，训练步数相同而非 FLOPs 相同；新 source 的多项改动尚未逐项消融。

## A：修正归一化及适配训练后

direct R² 为三个 seed 均值 ± population SD；none 无 seed 方差。

| 方法 | M1 | M2 |
|---|---:|---:|
| none | **0.673568** | 0.233793 |
| IO | 0.658046 ± 0.000170 | 0.272154 ± 0.002232 |
| projection LoRA | 0.650999 ± 0.005825 | 0.319075 ± 0.016364 |
| channel affine | 0.615414 ± 0.008108 | 0.299668 ± 0.001306 |
| affine + LoRA | 0.648637 ± 0.006187 | 0.324433 ± 0.024952 |
| rotated offset | 0.643893 ± 0.001815 | 0.315052 ± 0.008758 |
| original-coordinate offset | 0.641828 ± 0.001490 | 0.333497 ± 0.012924 |
| progressive full | 0.641207 ± 0.001599 | **0.345628 ± 0.002798** |

M1 旧骨架在 support z-score 后不需要监督适配；增加自由度降低 query 表现。M2 仍有剩余漂移，full 对 none 的平均收益为0.1118，三个 seed 方向一致。affine+LoRA 并未成为通用最优。

## B：session预训练、连续上下文与容量

| 任务/骨架 | none | IO | LoRA | affine + LoRA | original offset |
|---|---:|---:|---:|---:|---:|
| M1 old | 0.670773 | 0.642537 ± .007232 | 0.636240 ± .003619 | 0.634068 ± .004150 | 0.628282 ± .004892 |
| M1 small | 0.692343 | 0.720758 ± .000980 | **0.722944 ± .001767** | 0.722502 ± .001861 | 0.717219 ± .000996 |
| M1 wide | 0.691231 | 0.723933 ± .001375 | **0.733201 ± .000791** | 0.732981 ± .000793 | 0.713568 ± .002898 |
| M2 old | 0.216524 | 0.296406 ± .000277 | **0.333822 ± .008853** | 0.314631 ± .028978 | 0.266039 ± .010806 |
| M2 small | 0.369642 | 0.366213 ± .004489 | 0.386640 ± .014002 | **0.398763 ± .011241** | 0.389889 ± .001173 |
| M2 wide | 0.436734 | 0.460176 ± .002136 | 0.486414 ± .010595 | 0.477305 ± .004846 | **0.491188 ± .002023** |

新大模型对旧连续窗口 source 的配对收益：M1 LoRA +0.09696±0.00417，M2 original offset +0.22515±0.01250。新 source 含多个共同改动，不能单独归因于 session embedding 或漂移增强。

补充 M2 progressive full 为 small 0.380401、wide 0.464195（各三个seed），均未超过 wide LoRA/offset。补充使用独立 root，未改变冻结 B 矩阵。首次启动因 GPU 环境变量为整数而在训练前退出，零fit/零评分；修正启动器后在新root完成六fit、六评分，保留失败证据。

| source | 部署骨架参数 | 额外session训练参数 | fit秒 | GPU peak MiB |
|---|---:|---:|---:|---:|
| M1 small | 362,272 | 768 | 38.28 | 688.75 |
| M1 wide | 2,750,544 | 1,152 | 51.21 | 947.09 |
| M2 small | 364,562 | 1,920 | 45.07 | 435.83 |
| M2 wide | 2,755,138 | 2,688 | 60.73 | 693.15 |

small 参数减少约86.8%。M1 wide 最佳均值比small只高0.01026，参数约7.6倍；M2最佳均值则差0.09242。M2 wide LoRA 只训练37,000参数（约1.34%），可合并到普通投影；offset稍好；当前CPU部署支持plain及独立实现的unmerged LoRA，不支持state-offset。

GPU fit时间和allocated memory不能替代容器延迟或总进程RSS。source-val使用训练期session前端；普通均值前端val另有重放证据，不能混用。

## 官方预算与候选

M1补充严格使用前10个原始trial，允许此前neural-only数据参与输入统计，前导标签全部无效。M2严格使用前33个原始trial，保留短trial而不补足预算。loader独立核验全部7个M1、13个M2公开校准文件的原始表、统计和roster，M2共保留43个短trial。公开held-out calibration与hidden test标签属于不同数据面。

M1原始M10探针在old affine+LoRA seed2的7,700/8,000步峰值处拒绝，七fit、零query后停止，没有绕过收敛门槛。IO-only修订在读取任何query前排除该方法，并冻结完整失败root哈希；old/small/wide各执行none与三个IO seed，共12fit/12评分，独立验收通过。

| M1，10原始trial | none | IO mean ± SD |
|---|---:|---:|
| old | 0.673078 | 0.622328 ± 0.001329 |
| small | **0.688083** | 0.663260 ± 0.056962 |
| wide | 0.687293 | 0.696100 ± 0.001750 |

首个M1候选为small/none：wide IO仅多约0.0080，参数约7.6倍；small IO三个结果0.5827/0.7028/0.7043，不能用最高query seed代表方法。small/none不训练target权重，只用合法calibration神经统计，输出统计固定。七个session导出权重均逐tensor等同于B small source。

M2官方33-trial探针使用wide LoRA，LR0.003由A prefix选择固定转移；三个seed全部完成拟合才读取本地query。部署seed固定为0，不按最高query seed选择。最终状态与精确数字见末节。

本地query用于本轮架构探索比较，但没有用于矩阵内部checkpoint/LR选择。`combined_contract.json`的`query_used_for_selection=false`描述后者，不代表完全未查看本地开发query。隐藏评测只提交固定候选，不作hidden-score超参搜索。

## CPU部署及数值验收

`mamba3_cpu.py`以纯PyTorch复现pinned官方Mamba-3 SISO的BF16 chunk边界，范围限于本轮plain merged几何；独立的`mamba3_cpu_lora.py`支持固定rank4、alpha4、all-scope未合并LoRA。这是数值近似，不宣称逐bit相同。官方commit为`e9594ce1c732d97440f0332fdc43170a2294dbfa`，Apache-2.0许可证随payload保留。

M1 small/none完整50,591点：保存官方GPU R²为0.68808325，实际CPU为0.68808794，差+0.00000469，小于预设0.001门槛。两侧原始物理标签与索引逐元素一致，未来输入扰动不改变CPU prefix。点级最大差约0.01759，R²相近不等于全部输出相同。

真实FALCON1.0.2 SDK完成M1四个minival session的本机及CPU Docker评估：预测、mask、GT逐元素相同，评分差0。容器无网络、2CPU/4GiB、exit0、无OOM；payload全文件从不可变image ID读回验证SHA。minival仅验证接口，不代表hidden held-out。

`SSMFalconDecoder`按官方tag加载校准bank，固定z-score及source输出逆变换。每个recording有独立128-bin历史，`observe`推进时钟，trial done不清除历史，inactive padding行冻结。没有EMA、ridge或M2额外`/5`缩放。未知roster、错误batch cap、哈希漂移立即拒绝。

EvalAI控制器默认只读，显式`--execute`才推镜像及创建private submission。它核对sustechhku/team42279/test4599、配额、活动任务、候选哈希及image ID，创建fsync持久intent，阻止不确定POST的盲目重试。私有held-out回执及分数留在本地忽略目录`results/official_evalai/`，不公开到GitHub。

## 独立审核与原文修正

1. 旧EMA重复纳入support且用更新后的均值算方差。本轮prefix冻结到support结束，其后先预测再更新：`v'=(1-alpha)*(v+alpha*delta²)`。固定half-life3,000bin（50Hz下60秒），单列为无标签测试时适配。旧source的fixed support M1/M2为0.6735682/0.2337932；strictly-past EMA为0.6992543/0.2669104。提交候选不使用EMA。
2. 旋转换算修正为`(R(+Phi)q)^T h'=q^T R(-Phi)h'`。original-coordinate用不旋转的normalized C+bias，其他规模及BF16边界一致。
3. 新archive保留原始物理标签；旧normalize/denormalize往返可能产生不同truth hash。验收要求新archive按索引与原始NWB标签逐元素一致。
4. 冻结A/B receipt中的`root_input_bias_trainable=false`描述错误：实际IO/full输入bias可训练。真实parameter paths、数量、optimizer groups及变更hash已核验，描述错误不影响性能。保留冻结代码证据，新部署receipt使用正确描述。

训练前独立审核八种方法的官方GPU前向、step-zero、非零有限梯度、冻结参数及因果性；训练后核验checkpoint/normalizer/source/runtime/target/hash/cohort，并由保存数组float64重算R²，不依赖训练脚本summary。

M2 trial内[10,25)仍占约82.45% query SST；wide LoRA该段预测/真值方差比约0.476，original offset约0.593，幅度收缩尚未解决。legacy tail方差很少，继续仅作诊断。

## 证据与复现

- 正式A：`results/cross_session_iteration_a/official_20261003`；分析`results/analysis/cross_session_iteration/stage_a/`。
- 正式B：`results/cross_session_iteration_b/official_20261003`；分析`stage_b/`。
- 配对/成本/图：`comparison.csv`、`source_resources.csv`、`combined_contract.json`、`adaptation_comparison.png`、`capacity_comparison.png`。
- 时间分解：`temporal_b/`；独立审核：`review/FINAL_REVIEW.md`及其引用CSV/JSON。
- M1预算：`official_m1_m10_failed/`、`official_m1_m10_io_probe/`；M2 full：`m2_full_supplement/`。
- CPU：`ssm_decode/mamba3_cpu.py`、`falcon_decoder.py`；校准/导出：`official_calibration.py`、`official_payload.py`；容器：`deployment/falcon_cpu/`。

训练与离线校准沿用相邻`APST/src`的`apst.data.load`读取器，正式产物记录其文件哈希；数据根目录为`/mnt/data/work_host/SPINT/SPINT-main/data`。官方GPU参考需pinned Mamba、隔离Triton3.5.0及Torch2.5.1 CUDA环境。CPU推理镜像只依赖FALCON SDK和PyTorch CPU，不依赖APST或CUDA。

```bash
cd /home/xinyuan/Work_host/SSM
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/.tools/mamba_deps:$PWD" \
  /home/xinyuan/miniconda3/envs/spint/bin/python scripts/run_cross_session_iteration.py \
  --matrix configs/cross_session_iteration_a.json \
  --output-root results/cross_session_iteration_a/official_20261003 --phase all

PYTHONNOUSERSITE=1 PYTHONPATH="$PWD" \
  /home/xinyuan/miniconda3/envs/spint/bin/python scripts/summarize_cross_session_iteration.py \
  --root results/cross_session_iteration_a/official_20261003 \
  --output results/analysis/cross_session_iteration/stage_a --require-complete
```

## 未建立的结论

只有一个target/session及一个source seed；适配seed的SD不是跨target置信区间。C阶段多target轮换、时间前向划分及8/16/33/66 support曲线尚未完成。dt/A/B/C/angle分行微调、source预训练加入逐层offset、闭式无标签offset估计也未完成。后续应先用多个target复核本轮候选，再比较内部适配轴。

## 官方预算探索完成记录

M2合法33-trial探针完成4个fit、4次评分并通过独立验收。source-only为0.430611；LoRA三个seed为0.505119/0.485418/0.455886，均值0.482141、population SD0.020232，全部高于同预算none。每个fit严格使用33个原始trial（包括该target的6个短trial），固定LR0.003；query为与正式B相同的14,115个有效点。精确证据在`official_m2_m33_probe/`及`review/m2_m33_probe_audit.json`。

M2原始merged公开bank导出在第8个session Run1_20201030停止：已收敛（best400/2000），fold R²差仅0.00003779，但点级allclose失败，最大误差0.02036。该失败root保留，未放宽旧门槛。新root使用未合并的base+delta两次投影运算，复用已有prefix选定权重并完成余下bank；13个bank均通过独立原始预算、收敛、冻结参数和全部prefix-val点GPU/CPU R²差≤0.001检查。

固定M2 seed0的完整14,115点：官方GPU unmerged R²0.50513086，CPU0.50509506，差−0.00003580；unmerged GPU相对原merged参考只差+0.00001214，未改变候选选择或权重。部署未合并时保留LoRA额外参数及第二次投影，CPU成本须由SDK容器实测，不能沿用合并版延迟。

M1、M2候选均已使用sustechhku完成私有提交。M1官方评测已结束；M2已进入官方队列，等待评分。私有回执、提交ID及分数仅保留在本地忽略目录，不写入此公共文档。

控制器新增显式`--allow-active-intent PATH`：仅允许与已确认私有回执严格匹配、且不同于当前候选的活动提交；未指定时仍阻止任何活动提交。该改动通过独立审核及57项离线测试。实际M2提交发生在M1完成之后，沿用默认串行门禁；并行入口因团队锁退出，没有创建重复提交。审核记录见`review/private_controller_active_intent_audit.json`。
