<p align="center">
  <a href="README.md"><b>English (README.md)</b></a> · <b>简体中文 (README.zh-CN.md)</b>
</p>

> [!IMPORTANT]
> **实验性老硬件极限压榨分支声明（`gfx906` Vega 20 + `Broadwell` AVX2）**
>
> 本仓库是基于 **[Niko1221/Strata](https://github.com/Niko1221/Strata)** 的**实验性底层算子与异构协同极限优化分支**，目标是在一台总造价仅 **约 1,500 元人民币（≈\$207 USD）的二手老工作站**（`Intel Xeon E5-2686 v4` + `64 GB` 四通道 `DDR3-1866 ECC` + 双风扇改装 `AMD Instinct MI50 16GB HBM2`）上，彻底榨干硬件物理上限，流畅运行 **125B 级稀疏 MoE 大模型 `Qwen3.8-Flash-Next` (`GSQ-RCO-IQ2_XS`)** 并打穿 **256K（`262,144`）多轮 Agent 极限上下文**。
>
> 如需获取官方桌面安装包、跨平台发行版及多品牌显卡通用支持，请直接访问上游主仓库：**[github.com/Niko1221/Strata](https://github.com/Niko1221/Strata)**。

---

## 实验平台详细参数与极限实测战报（≈1,500 CNY 主机）

### 1. 测试平台完整硬件、固件与系统底座参数

| 组件维度 | 详细规格与底层调优状态 |
| :--- | :--- |
| **整机装机成本** | **约 1,500 元人民币（≈\$207 USD）**（X99 洋垃圾平台 + 二手计算卡改装方案） |
| **处理器 (CPU)** | **Intel Xeon E5-2686 v4**（14nm Broadwell-EP，**18 物理核 / 36 线程**，`performance` 调速器锁全核满睿频 **`3.00 GHz`**，支持 `AVX2` + `FMA3`；通过 `MSR 0x620 = 0x1b1b` 将 Uncore 环形总线/L3 锁满 **`2.70 GHz`**；**`45 MiB` 20-way L3 缓存**开启 `/sys/fs/resctrl` 硬件 CAT 物理隔离：**`18 ways (40.5 MiB)`** 独占给 CPU `0–17` AVX2 工作池，**`2 ways (4.5 MiB)`** 隔离 CPU `18–35` 后台 DMA 线程） |
| **系统内存 (RAM)** | **64 GB (`4 × 16 GB`) 四通道 DDR3-1866 ECC REG**（理论峰值带宽 `59.7 GB/s`；启用 `Transparent Huge Pages [always]` + `cudaHostRegister(PORTABLE)` 将 **`33.02 GiB`** 稀疏专家池与 **`2.39 GiB`** 256K QSA 流式 KV 池零拷贝锁页映射） |
| **加速卡 (GPU)** | **AMD Instinct MI50 16GB HBM2**（刷入 **AMD Radeon Pro VII** vBIOS，`Vega 20` / `gfx906:sramecc+:xnack-`，7nm，**60 CU / 3,840 流处理器**，Wave64，每 CU `64 KiB` LDS，**无任何 Matrix / Tensor Core**，具备硬件 `v_dot4_i32_i8`、`26.8 TFLOPS` 原生 FP16、1-cycle VALU DPP 与异步全局直灌 LDS `buffer_load_dword ... lds` DMA；Linux 运行时软注入 `pp_table` 零刷写硬件 SPI 解锁 **`1700 MHz` SCLK / `1100 MHz` HBM2（`1,126.4 GB/s`）/ `1240 MHz` FCLK**，全程 **`0` 次 UMC/GFX RAS ECC 纠错**，AST BMC 核显输出，实测可用显存 **`15.957 GiB`**） |
| **双风扇散热改装** | 被动散热器加装双风扇导风罩 + 自研 `hwmon` 守护服务（`strata-mi50-guard`）：连续满载 256K 预填充与长程 Agent 负载下稳控 **`32–48 °C` 边缘 / `42–60 °C` HBM2 / `47–84 °C` 结温**（全程 **0 次热降频**，`ue: 0, ce: 0`） |
| **PCIe 互连通道** | **PCIe 3.0 x16**（内核开启 `iommu=pt` 直通，`setpci` 将 BIOS 默认 `256B` 解锁至 **`MaxReadReq = 4096 Bytes`**，强制关闭根端口与显卡 ASPM L1）→ **实测持续 H2D 锁页 DMA 带宽达 `13.3 GB/s`** |
| **NVMe 存储流** | PCIe NVMe SSD 通过无缓冲 `O_DIRECT` 随机读取 `26.8 GiB`（3.2 亿行）逐层词表嵌入（PLE）n-gram 分片，实测 **`38,069 rows/s`** |
| **系统与编译栈** | Ubuntu 24.04 LTS（Linux 6.8，`ulimit -l unlimited`，`vm.swappiness=1`），宿主机原生提取 **`rocm-gfx906:7.14`**（`AMD clang 23.0.0git +PATCHED`、`rocBLAS 5.5.0.cd957402309`、`-O3 -march=broadwell -mtune=broadwell`、`CMAKE_HIP_ARCHITECTURES=gfx906`） |
| **评测大模型** | **ISTA-DASLab `Qwen3.8-Flash-Next` (`GSQ-RCO-IQ2_XS`)**（≈125B 总参数稀疏 MoE，每层 512 专家选 Top-10，36 层 GDN 线性递归 + 12 层 QSA 稀疏块注意力，`36.5 GiB` 主分片 + `26.8 GiB` PLE 分片 + `952 MiB` MTP 投机层，原生 `262,144` 训练上下文） |

---

### 2. 端到端实测成绩单

#### 2A. 严格同口径端到端 E2E 终版对比（原版冻结基线 vs 本仓库全栈优化终版）

下表每一行均采用**完全相同的 Prompt、随机种子与生成长度**，严格对比未修改的原版引擎构建（`build-baseline`，仅开 `32K` 上下文上限*；64K–256K 原版未跑项留空 `—`）与本分支纯软件栈 + 系统/GPU 内核级全栈优化终版（`build-opt`，零硬件编程器刷写）：

| 评测场景（`Qwen3.8-Flash-Next` `IQ2_XS` + MTP `--spec 4`） | 原版冻结基线 (`Baseline`, 仅 `32K` 上限)* | 本仓库全栈终版 (`build-opt`) | 严格同口径 E2E 增幅 | 精度校验与说明 |
| :--- | :---: | :---: | :---: | :--- |
| **日常对话 / 代码生成 Decode（9 次中位数 / 均值）** | `36.60 tok/s`（均值 `39.77`） | **`55.60 tok/s`**（均值 `59.98`） | **`+51.9%`** | 3 轮 9 次生成（**满配 `256K` 上限下中位数仍达 `56.90 tok/s`**） |
| **代码生成 Decode（线性投机峰值 `--spec 4`）** | `48.80 tok/s` | **`71.60 tok/s`**（热态重构 `76.3–77.8`） | **`+46.7%`** | 结构化代码高 MTP 接受率 + 高优先级 MTP 硬件流 |
| **代码生成 Decode（树状投机验证 `--spec-tree 3 --suffix-draft 2`）** | `48.80 tok/s` | **`75.40` / `83.80 tok/s`** | **`+54.5%` / `+71.7%`** | MTP Top-2 + Suffix 双分支因果树同窗验证（`merge` / 热态重构） |
| **零跳专家严格无损对话 (`Exact_Max`, `TAIL_SKIP=0`)** | `35.80 tok/s` (`E0`) | **`55.20 tok/s`**（均值 `56.92`，峰值 `62.2`） | **`+54.2%`** | CLI 4K + MTP 128-token 贪婪解码 **100% 逐位一致**（`47.21 tok/s`） |
| **零跳专家严格无损 4K Decode (`TAIL_SKIP=0`)** | `31.50 tok/s` (`E0`) | **`47.80` / `48.60 tok/s`**（冷 / 热） | **`+51.7%`（冷态）** | 纯无损 GCN 裸汇编 LDS-DMA + VALU DPP + AVX2 + 系统栈收益 |
| **4K 上下文 Prompt Prefill（冷态 / 热专家池）** | `308.30 tok/s` | **`550.70` / `614.00 tok/s`** | **`+78.6%` / `+99.2%`** | 热态预填充逼近 **`2.0x`** 翻倍（`I=128` MMQ + `rocBLAS` + `gfx906` PA） |
| **4K 上下文长文 Decode（冷态 / 热专家池）** | `35.50 tok/s` | **`54.20` / `49.80 tok/s`** | **`+52.7%`（冷态）** | `resctrl` L3 CAT 根治后台 DMA 污染（解码抖动标准差锐降 **16 倍**） |
| **32K 长上下文 Prefill（`27,773` tok，`int8` KV）** | `345.70 tok/s` (`80.4 s`) | **`666.00 tok/s` (`41.7 s`)** | **`+92.7%`** | 单次直省 **`38.7 秒`**；隐藏口令 100% 精准召回 |
| **32K 长上下文 Decode（`27,773` tok，`int8` KV）** | `43.90 tok/s`† | **`50.60 tok/s`** | **`+15.3%`** | 原版因摘抄原文触发高 Suffix 命中†；非复述真实基线参考 4K 行 `35.5 tok/s` |
| **64K 长上下文 Agent（`64,412` tok，`256K` `--kv-resident 16384`）** | `—` | **`656.70 tok/s` (`98.1 s`) / `48.20 tok/s`** | `—` | **满配 `262,144` 上限下仍驻留 `7,499` 专家槽（`10.04 GiB`）**；`54.3K` 达 `50.5–51.1 tok/s` |
| **248,320 全词表 FP64 前向 KL 散度** | `0.009351` | **`0.006085`（Top-1 `5/5` 一致）** | **KL 误差再降 `34.9%`** | `Exact_Max` KL `0.002010`；SpecPilot `5,760` 词延续 Top-1 吻合 **`99.27%–99.38%`** |

<sub>*注 1（为什么原版基线看起来已经不慢，且开了 256K 会暴跌）：上表 `Baseline` 实际已享受到 Step 0 宿主机底层调优红利（解锁 PCIe `4096B` `13.3 GB/s` DMA、`THP [always]` 零拷贝锁页与 `1000 MHz` 锁频，裸系统默认 BIOS `256B` 下更慢），且**仅开了 `--max-context 32768`**（显存得以塞入 `7,518` 个热专家槽）。原版一旦直接开启 `--max-context 262144`：默认 `int8`/`f16` KV 在 16GB 显存上直接 OOM；即便切到 `--kv k8v4` 静态预分配，由于缺乏本分支的 `--kv-resident` 锁页内存流式 KV 与 `mtp.cpp` 稠密环修复（多占 `263 MiB`），显存常驻热专家会暴跌超 **2,000 槽**（`7,518 → 5,489` 槽），真实非复述长上下文 Agent 生成会直接跌落至 **`~25–32 tok/s`** 区间。<br>†注 2（关于 32K 大海捞针行的 `43.90 tok/s`）：该题仅简短输出 `64` 个词并原样引用了档案原文词组，日志实锤原版触发了 `suffix drafts: 25 of 29 accepted`（超 40% 输出词直接靠原文 n-gram 字符串匹配白嫖）；原版在非原文摘抄场景下的真实长文生成速度请参考上方 4K 总结行（**`35.50 tok/s`**，严格无损 `31.50 tok/s`）。</sub>

#### 2B. 64K → 256K 超长上下文扩展与多轮 Agent 极限能力压测（单卡 16GB MI50）

下表为本优化分支在 `--kv k8v4` 下使用非重复多模块 C++20 代码库与长程运维日志实测的极限扩展表现（64K–256K 原版未跑项标 `—`）：

| Agent 阶段与活跃上下文深度 | 原版基线 | 显存常驻热专家槽 (`build-opt`) | Prefill 速率 / 增量 TTFT | Decode 速率 | 检索 / 推理 / 工具调用实测表现 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **64K Agent 摄入（`64,412` tok，`--kv-resident 16384` @ 256K 上限）** | `—` | **`7,499` 槽 (`10.04 GiB`)** | **`656.7 tok/s`**（冷预填充 `98.1 s`） | **`48.2 tok/s`**（`54.3K` 达 `50.5–51.1`） | 口令 100% 召回；QSA 稀疏块注意力 **`92.4%` 直接命中显存** |
| **128K 满载代码库冷摄入（`127,949` tok，静态 KV）** | `—` | `5,489` 槽 (`7.36 GiB`)* | **`484.3 tok/s`**（冷预填充 `264.2 s`） | **`38.9 tok/s`**（纯 MTP） | `19.1K` 与 `108.2K` 双深度密钥 **2/2 精准命中** |
| **256K 极限冷摄入 Turn 1（`251,964` tok，静态 KV）** | `—` | `5,489` 槽 (`7.36 GiB`)* | **`444.9 tok/s`**（冷预填充 `566.4 s`） | **`35.6 tok/s`**（纯 MTP） | `12.4K / 125.4K / 238.7K` 首中尾密钥 **3/3 一字不差** |
| **256K 多轮缓存续写 Turn 2（`255,042` tok，静态 KV）** | `—` | `5,489` 槽 (`7.36 GiB`)* | **增量 TTFT `9.63 s`**（新增 `2,619` 词 @ `271.8 tok/s`）** | **`54.8 tok/s`**（Suffix+MTP） | 命中引擎原生 Prefix Cache（复用 `252,163` 词）并续写 C++20 补丁 |
| **256K `<think>` + 工具调用（`255,693` tok，静态 KV）** | `—` | `5,489` 槽 (`7.36 GiB`)* | **`444.9 tok/s`**（冷预填充 `574.2 s`，0% 热降频） | **`41.5 tok/s`**（`<think>`+JSON） | 跨 **24.3 万词**将首尾双密钥精确填入 `run_patch_verification` 调用 |

<sub>*注 1：上表 128K–256K 静态 KV 冷摄入压测完成于 Step 6 之前；启用 Step 6（`--kv-resident 16384` + `mtp.cpp` 稠密环显存修复）后，开启完整 `262,144` 上下文时的显存常驻热专家容量已从 `5,489` 槽（`7.36 GiB`）跃升至 **`7,499` 槽（`10.04 GiB`）**。<br>**注 2：Turn 2 的 `9.63 s` 增量响应使用的是 Strata 引擎自带的会话 Prefix Cache 机制（原版同样支持），展示的是 25.5 万词水位下追加 ≈2.6K 工具报错时的实际多轮交互延迟。</sub>

---

### 3. 全栈逐层优化拆解（每一步是怎么榨出性能的）

1. **Step 0 — 宿主机 DMA、Uncore `2.70 GHz`、Intel RDT `resctrl` L3 硬件隔离与软注入 `pp_table`（`1,126 GB/s` HBM2，零刷硬件 BIOS）**
   - 解锁 PCIe `MaxReadReq=4096B` + `iommu=pt` 跑满 `13.3 GB/s` 持续 H2D DMA；`THP [always]` 锁页映射 `33.02 GiB` 专家池；通过 `MSR 0x620 = 0x1b1b` 将 Broadwell Uncore 环形总线锁满 **`2.70 GHz`**；启用 `/sys/fs/resctrl` 硬件 L3 CAT 将 `45 MiB` 20-way L3 物理切分为 **`18 ways`（CPU `0–17` AVX2 计算独占）与 `2 ways`（CPU `18–35` 后台 DMA 隔离）**，令 Decode 抖动标准差锐降 **16 倍**（`1.79 → 0.11 tok/s`）；并在 Linux 内核态软注入 `pp_table` 解锁 **`1700 MHz` SCLK / `1100 MHz` HBM2（`1,126.4 GB/s`）/ `1240 MHz` FCLK**，配合双风扇 `strata-mi50-guard` 实测 `ue: 0, ce: 0` 零 ECC 报错。
2. **Step 1 & 2 — Vega 20 (`gfx906`) 零寄存器溢出 Wave64 算子、`I=128` MMQ 与 Broadwell 双行/四行 AVX2 位平面池**
   - **GPU MoE 融合核与 Prefill MMQ (`iq_kernels.cu`, `mmq_moe.cu`)**：打通 `IQ2_S / IQ2_XXS / IQ1_M` Mode-8 单次 Launch 融合核（`Gate+Up+SwiGLU+Q8_1`），首创 **11-dword 无 Bank 冲突预转置位平面核**（`gcd(11,32)=1`），用 **`nib_neg4` 零分支补码取反**将 `v_dot4_i32_i8` 指令数腰斩（`16 → 8`），动态收敛 LDS 使驻留跃升至 **5 Blocks/CU** 且全工程 **`0` VGPR Spill（零寄存器溢出）**，并新增 `I=128` 两趟排序 MoE MMQ 分块核（`STRATA_MMQ_GROUP_SORT=1`）。
   - **路由链与 248K 大词表验证核 (`native_router.cu`, `verify_kernels.cu`)**：用 128-CTA 4 行分块核根治 `[512][2560]` Router 在 60 CU 上的尾波抖动（48 层路由链快 **`3.05x`**），Wave64 248K 大词表 `argmax_rows` 提速 **`7.84x`**。
   - **Broadwell 18 核 AVX2 位平面与双行专家池 (`q2_avx2.cpp`, `iq_avx2.cpp`)**：新增 4 行 `ymm` 寄存器分块核 `rows_bp4_nt1`（单核冷内存较 `ggml` 快 **`6.23x`**，17 核 `down` 阶段快 **`1.73x`**）以及 `IQ2_XS` 双行专家融合核复用 L1 `q8_1` 激活读取（CPU 专家池吞吐再涨 `+6–8%`）。
3. **Step 3 — 手写 GCN 裸内联汇编（`buffer_load_dword ... lds` 绕过寄存器异步直灌 LDS + 全引擎 1-Cycle VALU DPP）**
   - 在 `iq_kernels.cu` 与 `native_q2.cu` 中彻底绕过编译器 C++ 寄存器中转，手写构造 GCN 128-bit Buffer Resource Descriptor（`SRD`，`S_DATA_FORMAT=4` 硬件越界钳位）+ `m0` LDS 路由 + **`buffer_load_dword v0, v_off, s_srd, 0 offen offset:0 lds`**（ELF 反汇编实锤 **`308` 条硬件异步 Global-to-LDS DMA 指令**将 `q8_1` 激活直接从 HBM2 灌入共享内存，配合 `s_waitcnt lgkmcnt(0); s_barrier` 做到在途 `vmcnt` 权重预取零阻塞）；同时在 7 个核心算子文件（`verify_kernels.cu`, `native_bf16.cu`, `native_quant.cu`, `gdn_umv_tc.cuh`, `qsa_kernels.cu`, `mtp_kernels.cu`）中用 1-cycle 硬件 VALU DPP（`v_add_f32_dpp` / `v_mov_b32_dpp`）+ `ds_swizzle_b32` 替换跨线程 `ds_bpermute_b32`（在 Step 0–2 基础上再涨 `+6.2%` 对话与 `+10.6%` 4K Decode，且贪婪输出 **100% 逐位一致**）。
4. **Step 4 — CPU–GPU–PCIe 异构排期重平衡 (`pcie_frac=0.18`) + 高优先级 MTP 硬件流抢占 (`STRATA_MTP_HIP_PRIO=1`)**
   - 将未命中专家重平衡至 **`pcie_frac=0.18`**（82% 走 17 核 AVX2 池、18% 走 PCIe DMA）+ `spec_min_p=0.65` + Core 35 异步热专家换入（活跃对话显存命中率 **`93%–98%`**）+ `arch_defaults.cpp` 固化 17 项实测正向开关；并将 MTP 投机草稿生成挂载至高优先级硬件队列 `cudaStreamCreateWithPriority(..., leastPriority - 1)`（`STRATA_MTP_HIP_PRIO=1`），消除后台 PCIe 换入流排队抢占，将 MTP 起草耗时直接砍掉 **40%**（`5.24 → 2.95–3.35 ms/window`）。
5. **Step 5 — 随库附带 112 形状 `gfx906` `rocBLAS` FP16 表 + 手写 Wave64 `v_dot4_i32_i8` 在线 FlashAttention（32K Prefill 冲上 `666 tok/s`）**
   - 生成随库附带的 `tools/hip/gfx906-rocblas-5.5.0.cd957402309.txt`（GDN 主投影提速 **`2.05x–2.45x`**）配合 `STRATA_HIP_PROMPT_F16=1` + `STRATA_HC_UPMIX=1` 将 `hc down+silu` 耗时占比从 `26.3%` 压至 `3.4%`；并在 `qsa_prompt_attn.cu` 手写无矩阵核 Vega 20 专属 `prompt_attn_gfx906_kernel`（`4 × Wave64`、24-bit 三路拆分 `sdot4` 相对 FP64 误差仅 `1.7e-6`、融合 Sigmoid Gate 尾声、`Smem906` 互斥匿名 `union` 严控 `< 32 KB` LDS 达成 **每 CU 双 Block / 8 Wave64 驻留**），将 4K 热态 Prefill 推至 **`614.0 tok/s`**、32K–64K 长文 Prefill 推至 **`656.7–666.0 tok/s`**。
6. **Step 6 — 256K 锁页内存流式 KV (`--kv-resident 16384`) + `mtp.cpp` 稠密环显存修复（256K 常驻专家暴增 `+2,010` 槽）**
   - 修复 `src/model/mtp.cpp` 中 MTP 稠密环忽略 `kv_resident_cells` 多占显存的问题（单此一项立省 **`263 MiB`** 显存：`1,100 → 837 MiB`），并将 `262,144` 上下文的冷态 QSA `k8v4` KV 块放进 **`2.39 GiB` 零拷贝锁页 DDR3 内存**、每层仅在 HBM2 保留 `16,384` 热块（64K 深度下 QSA 块注意力 **`92.4%` 直接命中显存**）。在完整保持 `max_context=262144` 不缩水的前提下，显存常驻热专家从 `5,489` 槽（`7.36 GiB`）暴涨至 **`7,499` 槽（`10.04 GiB`）**，将 54K–64K Agent 长上下文 Decode 从 `~30 tok/s` 一举拉回 **`48.2–51.1 tok/s`**。
7. **Step 7 — 可选开启多分支因果树投机验证 (`--spec-tree 1/2/3`，代码生成冲上 `83.8 tok/s`)**
   - 打通端到端双分支树状投机草稿（设备端 MTP Step-0 Top-2 `d_top2_toks` + Suffix 分叉）与 48 层零溢出因果树并发验证：GDN 线性递归按 `tree_parent[t]` 精确分叉状态、QSA 稀疏注意力按 `tree_anc_mask[t]` 掩码可见祖先并在赢家分支确定后做 Clock-LRU 压缩、PLE n-gram 按树祖先哈希。经 SpecPilot `5,760` 词全词表 FP64 强制对齐严苛验证（`--spec-tree 3` Top-1 吻合率达 **`99.38%`**，高于默认线性 `--spec 4` 的 `99.27%`），结构化代码生成实测冲上 **`75.4–83.8 tok/s`**。

---

### 4. 在 `gfx906` + `Broadwell` 上一键复现构建与启动

```bash
# 1. 编译原生 gfx906 + Broadwell AVX2 极限引擎
cmake -B build-opt -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_HIP=ON \
  -DSTRATA_HIP_GFX906=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_CXX_FLAGS="-O3 -march=broadwell -mtune=broadwell"
cmake --build build-opt -j"$(nproc)"

# 2. 启动 256K Agent 生产服务（日常 32K 档可改用 --max-context 32768 --kv int8 --kv-grow --vram-reserve-mib 704；
#    重度结构化代码场景可按需追加 --spec-tree 3 --suffix-draft 2 开启多分支树状投机验证）
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export STRATA_Q2_BITPLANE=1 STRATA_CPU_IQ_ALL_EXACT1=1
export STRATA_ADAPT_OVERLAP=1 STRATA_ADAPT_JOB_CPU=35 STRATA_SPEC_COST=1
export STRATA_HIP_PROMPT_F16=1 STRATA_HC_UPMIX=1 STRATA_GFX906_PA=1
export STRATA_MMQ_GROUP_SORT=1 STRATA_MTP_HIP_PRIO=1
export STRATA_ROCBLAS_TUNING="$PWD/tools/hip/gfx906-rocblas-5.5.0.cd957402309.txt"

python3 serve/server.py --engine strata --host 127.0.0.1 --port 8090 -- \
  ./build-opt/strata --serve \
  --pack /path/to/packs/iq2_xs \
  --native /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --ple-gguf /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf \
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto \
  --spec 4 --spec-min-p 0.65 --mtp /path/to/mtp/rt \
  --max-context 262144 --kv k8v4 --kv-resident 16384 --pcie-frac 0.18 \
  --pool-workers 17 --host-core last --aux-cpus auto --vram-reserve-mib 576 \
  --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

<hr>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">在你自己的电脑上运行强大的 AI。</h3>
<p align="center">本地聊天、写代码、处理图片。</p>

<p align="center">
  <a href="#模型">模型</a> · <a href="#安装"><b>安装</b></a> ·
  <a href="#硬件">硬件</a> · <a href="#工具">工具</a> ·
  <a href="#帮助">帮助</a> · <a href="#支持-strata">支持</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="下载 Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT 许可证"></a>
  <a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff" alt="在 GitHub Sponsors 上支持 Strata"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="在 Buy Me a Coffee 上支持 Strata"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <b>简体中文</b> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

**免费、开源，由你自己运行。** 在桌面上使用 Strata，或搭配你喜欢的编程工具。无需账号。

## 模型

我们**首先推荐 ISTA-DASLab**，其次是 **Unsloth** 和其他提供方，前提是有内存放得下的兼容模型。

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| 提供方 | 可选项 |
| --- | --- |
| **ISTA-DASLab · 推荐** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S。[Coder：IQ1\_M](docs/MODELS.md#coder)，适合写代码。 |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)。占用内存更少，但速度更慢。 |
| **其他选项** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[比较大小、内存需求和兼容性](docs/MODELS.md)。

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0。这些是目前文档中收录的该模型的版本。

[模型指南和兼容性](docs/QWEN36.md)。

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

可在**桌面应用**和**终端**中使用。内存放不下的部分会从 SSD 读取，所以需要大量空闲磁盘空间。

| 版本（ISTA-DAS） | 下载大小 | 所需空闲磁盘 |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | 约245 GB |
| 2.00 bpw | 257 GB | 约265 GB |
| 2.25 bpw | 274 GB | 约280 GB |

支持 NVIDIA RTX 30、40 和 50 系列（Windows 和 Linux）、AMD Radeon AI PRO R9700（Linux）、AMD Ryzen AI Max — Strix Halo（Windows），以及内存 64 GB 或以上的 Apple 芯片 Mac（[Mac](docs/DEEPSEEK_MAC.md)）。安装程序会下载模型文件和 DeepSeek 引擎，然后启动模型。

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

以下为 Strata 0.2.5 上 2.00 bpw 版本的实测结果。“输出”是回答的速度，“提示词”是 Strata 读取你输入内容的速度。

| 硬件 | 输出 | 提示词 |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB 内存 | 4 到 6 tokens/s | 约 110 tokens/s（4K） |
| RTX 3060 12 GB, 46 GB 内存 | 9 到 11 tokens/s | 约 155 tokens/s（4K） |
| 1x Radeon AI PRO R9700, 251 GB 内存 | 21 到 25 tokens/s | 320 到 360 tokens/s (8K) |
| 4x Radeon AI PRO R9700 | 74 到 77 tokens/s | 400 到 450 tokens/s (8K) |

[内存、磁盘和详细信息](docs/DEEPSEEK41.md) · [发行说明](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

不确定哪个放得下？[为你的电脑选择模型](docs/MODELS.md#pick-by-memory)。

## 安装

<a id="option-a-the-strata-app"></a>

### 桌面应用

**[下载适用于 Windows、Linux 或 Mac 的 Strata →](https://github.com/Niko1221/Strata/releases/latest)**

安装应用，选择模型，就可以开始聊天。每个系统只有一个安装程序。首次启动时，应用会下载适合你显卡的引擎。[安装程序帮助](docs/INSTALL.md#the-app)。

<sub>文件: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata 桌面应用，聊天和实时代码预览并排显示">
  <br><sub>聊天、文件和实时预览都在一个窗口里。</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### 终端应用

更喜欢终端？安装 Strata，然后运行 **`strata launch`**。

<details open>
<summary><b>安装命令</b></summary>

**Linux / Mac**

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

在 Windows 上，安装后请打开一个新的终端。然后运行：

```sh
strata launch
```

按提示下载并启动模型。[终端指南](docs/TERMINAL_LAUNCH.md)。

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata 终端应用，显示模型活动、速度、内存和最近的请求">
  <br><sub>模型运行时，随时查看模型和硬件状态。</sub>
</p>

<a id="build-from-source"></a>

[从源码构建](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [完整安装指南](docs/INSTALL.md)

<a id="what-you-need"></a>

## 硬件

| 支持的硬件 | 指南 |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 系列** | [NVIDIA 安装](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900、RX 9060 / 9070、AI PRO R9700** | [AMD 安装](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · 实验性** | [RX 6000 安装](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel 安装](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 或更新** | [Mac 安装](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo 安装](docs/STRIX_HALO.md) |

<a id="using-it"></a>

## 工具

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**，也可以通过兼容 OpenAI 和 Anthropic 的 API 使用更多工具。

运行 **`strata connect`**，查看你所用工具的设置方法。[连接指南](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect)。

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## 帮助

[全部指南](docs) · [故障排查](docs/TROUBLESHOOTING.md) · [报告问题](https://github.com/Niko1221/Strata/issues/new/choose) ·
[基准测试](docs/COMMUNITY_BENCHMARKS.md) · [工作原理](docs/HOW_IT_WORKS.md) · [参与贡献](CONTRIBUTING.md)

## 支持 Strata

**帮助本地 AI 保持开放。** GitHub 赞助或一杯咖啡，都能帮助我们在更多硬件上开发和测试。

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors：支持 Strata" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee：支持 Strata" height="48"></a>

### 忠实支持者

感谢每一位支持 Strata 的人。

<!-- supporters:start (one line per public supporter, newest last; copy a line and change the GitHub name in all four places) -->
<a href="https://github.com/ryan4yin"><img src="https://github.com/ryan4yin.png?size=96" width="48" height="48" alt="ryan4yin" title="ryan4yin"></a>
<a href="https://github.com/themactep"><img src="https://github.com/themactep.png?size=96" width="48" height="48" alt="themactep" title="themactep"></a>
<a href="https://github.com/macintog"><img src="https://github.com/macintog.png?size=96" width="48" height="48" alt="macintog" title="macintog"></a>
<a href="https://github.com/bisonbet"><img src="https://github.com/bisonbet.png?size=96" width="48" height="48" alt="bisonbet" title="bisonbet"></a>
<a href="https://github.com/vizv"><img src="https://github.com/vizv.png?size=96" width="48" height="48" alt="vizv" title="vizv"></a>
<a href="https://github.com/kod88vn"><img src="https://github.com/kod88vn.png?size=96" width="48" height="48" alt="kod88vn" title="kod88vn"></a>
<a href="https://github.com/seoulworker"><img src="https://github.com/seoulworker.png?size=96" width="48" height="48" alt="seoulworker" title="seoulworker"></a>
<a href="https://github.com/jackyini"><img src="https://github.com/jackyini.png?size=96" width="48" height="48" alt="jackyini" title="jackyini"></a>
<a href="https://github.com/router0mail"><img src="https://github.com/router0mail.png?size=96" width="48" height="48" alt="router0mail" title="router0mail"></a>
<!-- supporters:end -->

[MIT 许可证](LICENSE) · [致谢和模型许可证](docs/HOW_IT_WORKS.md#credits) · [隐私](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>面向 AI 工具和自动阅读器： <a href="AGENTS.md">技术参考</a> · <a href="docs/AI_SETUP.md">安装说明</a></sub>

**赞助、硬件合作或其他合作：** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).
