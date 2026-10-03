# Real-2×2 SSM 固定点与硬件契约（2026-10-03）

此目录提供的是**可审计的软件位精确参考和单 mode RTL 骨架**，不是芯片测量、ASIC 面积估算或能耗主张。

## 整数递推

每个 real-`2×2` mode 的 state 为 `(p,q)`，接收整数 drive `u`，并以 int16 Q1.14 系数执行：

`[p';q'] = sat_W(round_nearest_ties_away((A_q14 [p;q] + B_q14 u) / 2^14))`。

- `A_q14=(a00,a01,a10,a11)` 和 `B_q14=(b0,b1)` 都是 signed int16 Q1.14；`u,p,q` 是无隐含小数位的 signed integer，取值单位由上游明确约定。
- 先以至少 64 bit 宽度累加所有乘积，**只在总和后一次**除以 `2^14`；正负半值均远离零（nearest, ties-away-from-zero）。随后对 `p'`、`q'` 各自饱和为 `STATE_W∈{16,24,32}`。绝不在整数核心调用 float。
- `saturation_count` 是每个被截断 state 加一的 32-bit 计数器；它是溢出诊断，不会阻止运算。`valid && ready` 每周期接受一笔；当前单周期实现 `ready=1`。共享 sequential-MAC 的资源优化可降低吞吐，但必须保留相同的“wide sum → one rounding → saturate”语义。
- Python `quantize_stable_matrix` 在量化后计算 `ρ(A)`，只有 `ρ(A)<1-margin` 才接受；直接构造 mode 时也拒绝 `ρ(A)≥1`。这是线性、零输入模型的必要保护，不取代有限字长的验证。

软件接口在 [fixedpoint.py](../ssm_decode/fixedpoint.py)，RTL 为 [real2x2_ssm_engine.sv](../hardware/real2x2_ssm_engine.sv)。`Real2x2ModeBank` 接受每 mode 的 `A_q14`（以及可选 `B_q14`）数组，mode 间没有隐藏耦合；`export_json()` 输出整数系数、后量化 spectral radius、state 及寄存器 bit layout；`hardware/generate_vectors.py` 使用相同参考生成 256 条确定性随机向量，testbench 按拍检查 `p/q/saturation_count`。

`[bank_w64_q14.json](../hardware/bank_w64_q14.json)` 从实际 `ModelConfig(96,2,width=64,kind='bank')` 导出全部 64 个 Q1.14 `A` 矩阵（shape `[64,4]`），分配 128 个 state values，并保留四种 codebook matrix。该软件 bank 的定义是 `nxt[...,0] += u`，所以桥接文件明确采用 `B=(16384,0)`；这与独立 RTL demo 向量为覆盖双坐标路径而采用的 `B=(0,16384)` 不同，二者不得混用。相同的 256 个 integer drive 只用于 recurrence 表征：int16 总计 4,656 次 state 饱和，int24/int32 均为 0；这不报告端到端 decoder score。

## 100k 零输入表征（不是证明）

`run_zero_input_characterization` 从给定初态步进最多 100,000 次，报告 peak state、饱和次数、重复 state（有限状态空间中的 limit-cycle 指示）及最终 state。测试使用 Q1.14 `A=[[15565,-2344],[2344,15565]]`、int24 初态 `(1,000,000,-700,000)`；期望结果是无饱和且 peak 不超过 1,100,000。阻尼旋转可以暂时重分配两个坐标，故 peak 不必小于初始单坐标峰值。这个结果仅是该配置、初态、舍入规则下的有限运行表征，不证明所有输入/系数都不会溢出或无极限环。

## 系统边界与校准存储

| 功能 | 合理映射 | 本骨架覆盖 |
|---|---|---|
| 静态输入/输出投影 | 数字 FPGA/ASIC MAC 的共享接口；也可在 RRAM crossbar 后以同一整数 `u` 接口交给本模块 | 否；只接收 integer drive `u` |
| real-`2×2` 递推、舍入、饱和 | FPGA/ASIC 的 SRAM 邻近数字 MAC / 寄存器；RRAM-CIM 路线也可保留为数字状态后端 | 是，单 mode |
| exact RLS `P∈R^{d×d}`、profile、频繁校准写入 | SRAM + 数字控制/除法或 reciprocal | 否；必须计入 `d²` state words 和每个带标签样本的 `O(d²)` 读写/MAC |

数字 FPGA/ASIC 与 RRAM-CIM 两条路线共用的接口是：上游给出 signed integer `u`，寄存器给出 `p/q`，并写入 Q1.14 `A/B`；`valid/ready` 表示一次状态提交。RRAM 路线若产生模拟投影，ADC/量化必须在这个接口之前完成；本模块不假定其实现方式。没有将 input-dependent selective decay、cross-attention、ADC/DAC、写耐久、阵列非理想性或 RLS 映射为已完成硬件，也没有任何能耗或面积估算。相关 CIM 文献可以支持“静态投影和部分固定递推可 co-design”的研究方向，不能构成本模块的实测芯片指标。

## 验证命令

```bash
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python -m pytest tests/test_fixedpoint.py -q
```

最后一个测试生成向量、编译 SystemVerilog，并将 RTL 每拍输出与 Python bit-exact 参考相比较。若系统没有 `iverilog`，测试会优先使用仓库局部的 `.tools/iverilog`（必须传 `-B .tools/iverilog/usr/lib/x86_64-linux-gnu/ivl`）；两者都没有时才 skip。
