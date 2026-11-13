<p align="center">
  <a href="README.md"><b>English (README.md)</b></a> · <b>简体中文 (README.zh-CN.md)</b>
</p>

> [!IMPORTANT]
> **老旧计算卡与服务器平台深度优化分支（`AMD MI50 16GB` + `Xeon E5-2686 v4`）**
>
> 本仓库是基于 **[Niko1221/Strata](https://github.com/Niko1221/Strata)** 的**硬件定向深度优化分支**，展示如何在总成本仅 **约 1,500 元人民币（≈\$207 USD）的二手工作站**（`Intel Xeon E5-2686 v4` + `64 GB DDR3-1866 ECC` + 双风扇散热改装 `AMD Instinct MI50 16GB HBM2`）上，流畅运行 **125B 稀疏 MoE 大模型 `Qwen3.8-Flash-Next` (`IQ2_XS`)**，并完整支持 **256K（`262,144`）超长上下文多轮 Agent 推理**。
>
> 如需下载官方桌面客户端或获取通用多显卡支持，请访问上游主仓库：**[github.com/Niko1221/Strata](https://github.com/Niko1221/Strata)**。

---

## 实验平台与端到端实测表现（≈1,500 CNY 主机）

### 1. 硬件配置与系统底座

| 组件 | 规格与系统配置 |
| :--- | :--- |
| **整机成本** | **约 1,500 元人民币（≈\$207 USD）**（二手 X99 平台 + 计算卡风冷改装） |
| **处理器 (CPU)** | **Intel Xeon E5-2686 v4**（14nm Broadwell-EP，**18 核 / 36 线程**，全核满频 **`3.00 GHz`** `AVX2` + `FMA3`；Uncore 环形总线锁频 **`2.70 GHz`**；`45 MiB` 20 路 L3 缓存通过 Linux `resctrl` 硬件切分：**`40.5 MiB`（18 路）** 留给 CPU 推理线程，**`4.5 MiB`（2 路）** 隔离后台 PCIe 传输线程） |
| **系统内存 (RAM)** | **64 GB (`4 × 16 GB`) 四通道 DDR3-1866 ECC REG**（理论带宽 `59.7 GB/s`；开启 2 MiB 透明大页 `THP [always]` 将 **`33.02 GiB`** 稀疏专家权重与 **`2.39 GiB`** 256K 流式 KV 缓存锁页映射，支持零拷贝读取） |
| **显卡 (GPU)** | **AMD Instinct MI50 16GB HBM2**（刷入 Radeon Pro VII 固件，`Vega 20` / `gfx906`，7nm，**60 CU / 3,840 流处理器**，**无 Tensor / Matrix 矩阵核心**，支持原生 `INT8` 点积、`26.8 TFLOPS` FP16 与 GCN 跨线程寄存器通信；通过 Linux 内核软加载功耗频率表运行于 **`1700 MHz` 核心 / `1100 MHz` HBM2（`1,126 GB/s`）**，无需编程器刷写硬件 BIOS，全程 **0 次显存 ECC 纠错**，可用显存 **`15.96 GiB`**） |
| **散热方案** | 双风扇导风罩改装 + 自动温控服务：持续满载预填充下维持 **`32–53 °C` 核心边缘 / `42–60 °C` 显存 / `47–87 °C` 结温**，全程零降频 |
| **PCIe 通道** | **PCIe 3.0 x16**（开启 `iommu=pt` 直通，将主板默认 `256B` 最大读取请求解锁至 **`4096 Bytes`** 并关闭节能休眠，实测持续主机到显卡 DMA 带宽 **`13.3 GB/s`**） |
| **固态存储** | PCIe NVMe SSD（通过 `O_DIRECT` 直接读取 `26.8 GiB` 词表嵌入分片，实测随机读取 **`38,069 行/秒`**） |
| **软件环境** | Ubuntu 24.04 LTS（Linux 6.8），原生编译 **`rocm-gfx906:7.14`**（`rocBLAS 5.5.0`，`-O3 -march=broadwell`） |
| **测试模型** | **ISTA-DASLab `Qwen3.8-Flash-Next` (`GSQ-RCO-IQ2_XS`)**（≈125B 总参数稀疏 MoE，每层 512 专家激活 10 个，36 层线性递归 GDN + 12 层稀疏块注意力 QSA，带 MTP 投机草稿层，原生支持 `262,144` 上下文） |

---

### 2. 实测性能对比

#### 2A. 原版 Strata vs 本分支优化终版（端到端同口径对比）

在同一台机器、相同测试集与输出长度下，原版 Strata 与本分支全栈优化终版的实测速度对比：

| 测试场景（`Qwen3.8-Flash-Next` `IQ2_XS`） | 原版 Strata | 本分支优化终版 | 端到端提升 | 说明 |
| :--- | :---: | :---: | :---: | :--- |
| **日常对话与代码生成（多轮中位数 / 均值）** | `36.6 tok/s`（均值 `39.8`） | **`55.6 tok/s`**（均值 `60.0`） | **`+51.9%`** | 开启 256K 上下文上限时中位数同样达 `56.9 tok/s` |
| **代码生成峰值（默认投机解码）** | `48.8 tok/s` | **`71.6 tok/s`**（热缓存 `76–78`） | **`+46.7%`** | 结构化代码高投机接受率 + 高优先级草稿队列 |
| **代码生成峰值（开启多分支树状投机）** | `48.8 tok/s` | **`75.4` – `83.8 tok/s`** | **`+54.5%` ~ `+71.7%`** | 同时验证多条候选分支（`--spec-tree 3 --suffix-draft 2`） |
| **100% 全专家无损对话解码** | `35.8 tok/s` | **`55.2 tok/s`**（峰值 `62.2`） | **`+54.2%`** | 关闭低权重专家跳过，128 词贪婪输出与原版 **100% 逐字一致** |
| **100% 全专家无损 4K 长文解码** | `31.5 tok/s` | **`47.8` / `48.6 tok/s`** | **`+51.7%`** | 纯算子与异构调度带来的无损提速 |
| **4K 上下文预填充 Prefill（冷态 / 热缓存）** | `308.3 tok/s` | **`550.7` / `614.0 tok/s`** | **`+78.6%` / `+99.2%`** | 热缓存下预填充速度接近翻倍 |
| **4K 上下文长文生成 Decode** | `35.5 tok/s` | **`54.2 tok/s`** | **`+52.7%`** | 硬件隔离 L3 缓存后，生成延迟波动降低 16 倍 |
| **32K 长上下文预填充 Prefill（`27.8K` 词）** | `345.7 tok/s` (`80.4 s`) | **`685.7 tok/s` (`40.6 s`)** | **`+98.4%`** | 耗时直接减半（省下约 `40 秒`），隐藏口令 100% 准确找回 |
| **32K 长上下文长文生成 Decode（`27.8K` 词）** | `~38.6 tok/s` | **`55.0 tok/s`** | **`+42.5%`** | 检索信息后连续原创生成 256 词技术长文 |
| **64K 长上下文预填充 Prefill（开启 `256K` 上限）** | `~331.7 tok/s` (`~194 s`) | **`655.7 tok/s` (`98.3 s`)** | **`+97.7%`** | 单次预填充节省约 `96 秒`，隐藏口令 100% 准确找回 |
| **64K 长上下文生成 Decode（开启 `256K` 上限）** | `~33.3 tok/s` | **`54.6 tok/s`** | **`+64.0%`** | 原版开 256K 显存专家缩水至 `5,539` 个；本版保住 **`7,499` 个**，相比 32K 零衰减 |
| **全词表输出分布误差（FP64 KL 散度）** | `0.00935` | **`0.00609`（Top-1 全一致）** | **误差降低 `34.9%`** | 不仅速度更快，数值精度也比原版默认配置更接近全精度基准 |

#### 2B. 64K → 256K 超长上下文与多轮 Agent 压测（单卡 16GB MI50）

在单张 16GB 显存的 MI50 上开启完整 `262,144` 上下文上限，使用非重复多模块 C++20 代码库实测从 64K 到 256K 的长文分析与多轮 Agent 表现（原版未跑项标 `—`）：

| 测试场景与活跃上下文长度 | 原版 Strata | 本版显存常驻专家数 | 预填充速度 / 多轮首字延迟 | 生成速度 (Decode) | 准确性与 Agent 任务表现 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **64K 长文档分析（`64,452` 词）** | `~331.7` / `~33.3 tok/s` | **`7,499` 个 (`10.04 GiB`)** | **`655.7 tok/s`**（冷启动 `98.3 s`） | **`54.6 tok/s`** | 口令 100% 找回；稀疏注意力 `92.4%` 直接命中显存 |
| **128K 完整代码库冷加载（`127,949` 词）** | `—` | **`7,499` 个 (`10.04 GiB`)** | **`646.4 tok/s`**（冷启动 `198.0 s`） | **`56.2 tok/s`** | 分布在 `19K` 与 `108K` 处的两把密钥全部准确找回 |
| **256K 极限代码库冷加载 Turn 1（`251,964` 词）** | `—` | **`7,499` 个 (`10.04 GiB`)** | **`614.5 tok/s`**（冷启动 `410.0 s`） | **`51.8 tok/s`** | 横跨首、中、尾（`12K / 125K / 239K`）三处密钥一字不差；`90.5%` 注意力块命中显存 |
| **256K 多轮对话续写 Turn 2（`255,037` 词）** | `—` | **`7,499` 个 (`10.04 GiB`)** | **首字延迟 `6.80 s`**（增量 `2,618` 词，`384.8 tok/s`） | **`55.3 tok/s`** | 复用会话前缀缓存（免重算前 `25.2 万` 词）并生成 C++20 补丁；`93.6%` 注意力块命中显存 |
| **256K 深度思考 + 工具调用 Turn 3（`255,653` 词）** | `—` | **`7,499` 个 (`10.04 GiB`)** | **`614.3 tok/s`**（冷启动 `415.8 s`） | **`55.9 tok/s`** | 跨 `24.3 万` 词提取多处密钥输出合法 JSON 工具调用；`94.9%` 注意力块命中显存 |

---

### 3. 核心优化步骤概览

1. **打通主板、CPU 缓存与显卡带宽底座（零硬件刷写）**
   - 解锁主板 PCIe `4096B` 最大读取请求（持续传输达 `13.3 GB/s`），开启 2 MiB 透明大页零拷贝映射 `33 GiB` 内存专家池；将 Xeon CPU 的环形总线（Uncore）锁满 `2.70 GHz`，并用 Intel `resctrl` 硬件缓存隔离技术把后台 PCIe 传输限制在 2 路 L3 缓存内，避免冲刷 18 核 CPU 计算所需的缓存（生成速度波动降低 **16 倍**）；在 Linux 启动时软加载显卡功耗频率表，安全解锁 **`1700 MHz` 核心与 `1100 MHz` 显存（`1,126 GB/s`）**。
2. **针对无矩阵核心的 Vega 20 与老至强重写底层算子**
   - MI50 没有现代显卡的 Tensor Core。我们针对其 Wave64 架构重写了融合 MoE 专家算子、无冲突共享内存位平面转置、路由矩阵乘法（提速 **`3.05x`**）以及 24.8 万词表并行采样（提速 **`7.84x`**），并确保全工程 **零寄存器溢出（0 VGPR Spills）**；同时在 CPU 端编写了专门的 `AVX2` 多行寄存器分块与双专家共享激活读取算子，使 17 核 CPU 专家计算提速 **`1.73x`**。
3. **手写 GCN 裸汇编：绕过寄存器直灌共享内存 + 单周期跨线程归约**
   - 进一步突破编译器生成代码的限制，在核心量化算子中手写 AMD GCN 内联汇编，利用硬件异步指令（`buffer_load_dword ... lds`）将输入激活向量**直接从 HBM2 显存异步搬运进片上共享内存（LDS）**，省去通用寄存器中转开销，并与权重读取完全重叠；同时在全引擎 7 大核心模块中用 1 周期硬件跨线程指令（VALU DPP）替换慢速共享内存交换，贪婪解码结果保持 **100% 逐字无损**。
4. **CPU–GPU–PCIe 协同负载均衡与高优先级投机草稿流**
   - 重新标定异构分工比例（82% 显存未命中专家交由 17 核 CPU 并行计算，18% 走 PCIe DMA 传给显卡），配合后台线程实时将高频专家换入显存（活跃对话显存命中率达 **`93%–98%`**）；同时将 MTP 投机草稿模型放入独立的高优先级硬件队列，避免被后台权重传输阻塞，使草稿生成耗时降低 **40%**。
5. **定制 112 组 `rocBLAS` 矩阵调优表与手写 FlashAttention（预填充近翻倍）**
   - 针对本模型的全部 112 种矩阵尺寸实测生成 Vega 20 专属 `rocBLAS` FP16 最优算子表，并为无矩阵核心的 MI50 手写了严格控制在 `32 KB` 共享内存以内的 `INT8` 点积版在线 FlashAttention（单计算单元可并发驻留双倍线程块），使 32K–256K 全程长文本预填充稳定跑在 **`614–685 tok/s`**。
6. **256K 锁页内存流式 KV 缓存（32K 到 256K 生成几乎零衰减）**
   - 修复了草稿层在流式模式下多占 `263 MiB` 显存的问题，并将 256K 上下文的冷态 KV 缓存放入 `2.39 GiB` 锁页系统内存，显存中每层仅保留最活跃的 `16K` 窗口（即使在 25.5 万词深度下，仍有 **`90.5%–94.9%`** 的稀疏注意力读取直接命中显存）。完整开启 `262,144` 最大上下文后，显存常驻专家数从 `5,489` 个大幅回升至 **`7,499` 个（`10.04 GiB`）**，从 32K 一路拉满到 256K，生成速度始终稳在 **`51.8–56.2 tok/s`**。
7. **多分支树状投机解码（代码生成冲上 `83.8 tok/s`）**
   - 新增可选的多分支因果树投机验证（`--spec-tree 1/2/3`）：当投机模型遇到高不确定性分叉时，在同一个推理步内沿着注意力掩码与递归状态同时验证两条候选分支，保留命中更长的一支，在代码编写与重构任务上最高可达 **`75.4–83.8 tok/s`**。

---

### 4. 编译与启动命令参考

```bash
# 1. 编译针对 gfx906 (MI50) 与 Broadwell (Xeon v4) 的优化引擎
cmake -B build-opt -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_HIP=ON \
  -DSTRATA_HIP_GFX906=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_CXX_FLAGS="-O3 -march=broadwell -mtune=broadwell"
cmake --build build-opt -j"$(nproc)"

# 2. 启动 256K 上下文生产服务（日常 32K 模式可改为 --max-context 32768 --kv int8 --kv-grow --vram-reserve-mib 704；
#    重度代码生成场景可按需追加 --spec-tree 3 --suffix-draft 2 开启树状投机解码）
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
