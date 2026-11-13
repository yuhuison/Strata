<p align="center">
  <b>English (README.md)</b> · <a href="README.zh-CN.md"><b>简体中文 (README.zh-CN.md)</b></a>
</p>

> [!IMPORTANT]
> **Experimental Hardware-Limit Optimization Fork (`gfx906` Vega 20 + `Broadwell` AVX2)**
>
> This repository is an **experimental research & kernel-optimization fork** of **[Niko1221/Strata](https://github.com/Niko1221/Strata)** dedicated to extracting the absolute physical ceiling of an **≈1,500 CNY (≈\$207 USD) legacy workstation** (`Intel Xeon E5-2686 v4` + `64 GB DDR3-1866 ECC` + dual-fan modded `AMD Instinct MI50 16GB HBM2`) running the **125B-class sparse MoE model `Qwen3.8-Flash-Next` (`GSQ-RCO-IQ2_XS`)** all the way up to **256K (`262,144`) multi-turn Agent context**.
>
> For official releases, desktop installers, and general multi-GPU/multi-vendor support, please use the upstream repository: **[github.com/Niko1221/Strata](https://github.com/Niko1221/Strata)**.

---

## Experimental Platform & Benchmark Showcase (≈1,500 CNY / ≈\$207 Rig)

### 1. Detailed Hardware, Firmware & System Specifications

| Component | Exact Specification & Tuning State |
| :--- | :--- |
| **Total Build Cost** | **≈1,500 CNY (≈\$207 USD)** (secondary-market X99 + Vega 20 compute build) |
| **CPU** | **Intel Xeon E5-2686 v4** (14nm Broadwell-EP, **18 Cores / 36 Threads**, `3.00 GHz` sustained all-core AVX2 + FMA3 under `performance` governor, `45 MiB` 20-way L3 cache with `/sys/fs/resctrl` CAT, single-socket NUMA node 0) |
| **System RAM** | **64 GB (`4 × 16 GB`) Quad-Channel DDR3-1866 ECC Registered** (`59.7 GB/s` theoretical peak; `33.02 GiB` sparse expert arena zero-copy pinned via `Transparent Huge Pages [always]` + `cudaHostRegister(PORTABLE)`) |
| **GPU** | **AMD Instinct MI50 16GB HBM2** flashed with **AMD Radeon Pro VII** vBIOS (`Vega 20` / `gfx906:sramecc+:xnack-`, 7nm, **60 CUs / 3,840 SPs**, Wave64, `64 KiB` LDS/CU, **NO Matrix / Tensor Cores**, native `v_dot4_i32_i8` + `26.8 TFLOPS` FP16 + 1-cycle VALU DPP; `16 GiB` 4096-bit HBM2 locked at **`1000 MHz` = `1,024 GB/s`**, `15.957 GiB` usable VRAM with AST BMC headless display) |
| **Thermal Mod** | Custom dual-fan shroud mod + `hwmon` PWM daemon (`strata-mi50-guard`): holds **`32–46 °C` edge / `42–60 °C` HBM2 / `47–84 °C` junction** across 25+ minutes of continuous 256K full-load prefill (`118.4 W` mean, zero thermal throttling) |
| **PCIe Interconnect** | **PCIe 3.0 x16** (`iommu=pt` passthrough, `setpci` unlocked **`MaxReadReq = 4096 Bytes`** from BIOS `256B`, ASPM L1 disabled) → **`13.3 GB/s` sustained H2D pinned DMA** |
| **NVMe Storage** | PCIe NVMe SSD streaming the `26.8 GiB` (`320M`-row) Per-Layer Embedding (PLE) n-gram shard via unbuffered `O_DIRECT` at **`38,069 rows/s`** |
| **OS & Compiler** | Ubuntu 24.04 LTS (Linux 6.8, `ulimit -l unlimited`, `vm.swappiness=1`), host-extracted native **`rocm-gfx906:7.14`** (`AMD clang 23.0.0git +PATCHED`, `rocBLAS 5.5.0.cd957402309`, `-O3 -march=broadwell -mtune=broadwell`, `CMAKE_HIP_ARCHITECTURES=gfx906`) |
| **Evaluated Model** | **ISTA-DASLab `Qwen3.8-Flash-Next` (`GSQ-RCO-IQ2_XS`)** (≈125B sparse MoE, 512 experts top-10, 36 GDN linear-recurrence + 12 QSA sparse block-attention layers, `36.5 GiB` Shard 1 + `26.8 GiB` PLE Shard 2 + `952 MiB` MTP speculative drafter, native `262,144` trained context) |

---

### 2. End-to-End Performance Scorecard

#### 2A. Strict Apples-to-Apples A/B Speedups (Frozen Upstream Baseline vs. Optimized Fork)

Every row below compares the unmodified upstream build (`build-baseline`) against this fork (`build-opt`) under identical prompt/cache conditions:

| Workload (`Qwen3.8-Flash-Next` `IQ2_XS` + MTP `--spec 4`) | Frozen Upstream (`Baseline`) | Optimized Fork (`build-opt`) | Measured A/B Gain | Verification & Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Interactive Chat / Code Decode (Median)** | `36.60 tok/s` | **`45.60 tok/s`** | **`+24.6%`** | 3-round counter-balanced confirmation (9 runs) |
| **Code Generation Decode (Peak)** | `48.80 tok/s` | **`57.00 tok/s`** (`55+` sustained) | **`+16.8%`** | High MTP 4-token acceptance on structured code |
| **Zero-Skip Exact Chat (`Exact_Max`, `TAIL_SKIP=0`)** | `35.80 tok/s` (`E0`) | **`46.20 tok/s`** (`45.27` mean) | **`+29.1%`** | CLI 128-token greedy **100% bitwise identical** |
| **Zero-Skip Exact 4K Decode (`TAIL_SKIP=0`)** | `31.50 tok/s` (`E0`) | **`40.40 tok/s`** (`42.6` peak) | **`+28.3%`** | Pure lossless MoE + CPU/GPU co-scheduling gain |
| **4K Context Prompt Prefill (Cold / Warm Profile)** | `308.30 tok/s` | **`426.50` / `455.20 tok/s`** | **`+38.3%` (cold)** | Warm reaches `455.2` (`463.6 tok/s` w/ `STRATA_PA_FAST=1`) |
| **4K Context Long Decode (Cold / Warm Profile)** | `35.50 tok/s` | **`40.10` / `43.30 tok/s`** | **`+13.0%` (cold)** | Warm adaptive cache reaches `43.30 tok/s` |
| **32K Long-Context Prefill (`27,773` tok, `int8` KV)** | `345.70 tok/s` (`80.4 s`) | **`523.30 tok/s` (`53.1 s`)** | **`+51.4%`** | Saves `27.3 s` per 28K prompt; 100% needle hit |
| **32K Long-Context Decode (`27,773` tok, `int8` KV)** | `43.90 tok/s` | **`47.30 tok/s`** | **`+7.7%`** | Zero degradation at 28K depth |
| **248,320 Full-Vocab FP64 Forward KL Divergence** | `0.009351` | **`0.009299` (`5/5` Top-1 match)** | **Lower KL error** | `Exact_Max` mean KL: **`0.000563`** (vs `E0_Golden_Exact`) |

#### 2B. 64K → 256K Extreme Context Scaling & Multi-Turn Agent Stress Test (Single 16GB MI50)

Measured on the optimized fork (`--kv k8v4`) using non-repetitive multi-module C++20 repository snapshots to verify ultra-long Agent capability on a single 16GB HBM2 card (upstream baseline not re-run at 64K–256K):

| Agent Stage & Context Depth | Resident VRAM Experts | Prefill / Incremental TTFT | Decode Speed | Accuracy / Agent Workflow Verification |
| :--- | :---: | :---: | :---: | :--- |
| **64K Cold Ingestion (`64,412` tok)** | `6,707` slots (`8.99 GiB`) | **`519.0 tok/s`** (`124.1 s` cold) | **`43.8 tok/s`** (pure MTP) | 100% needle passcode recall |
| **128K Full Cold Repo (`127,949` tok)** | `5,489` slots (`7.36 GiB`) | **`484.3 tok/s`** (`264.2 s` cold) | **`38.9 tok/s`** (pure MTP) | **2/2** audit keys (`19.1K` & `108.2K`) recalled verbatim |
| **256K Extreme Cold Turn 1 (`251,964` tok)** | `5,489` slots (`7.36 GiB`) | **`444.9 tok/s`** (`566.4 s` cold) | **`35.6 tok/s`** (pure MTP) | **3/3** keys (`12.4K / 125.4K / 238.7K`) recalled verbatim |
| **256K Prefix-Cached Turn 2 (`255,042` tok)** | `5,489` slots (`7.36 GiB`) | **TTFT `9.63 s`** (`271.8 tok/s` on `2,619` new tok)* | **`54.8 tok/s`** (Suffix+MTP) | Reuses Strata's native prefix cache (`252,163` tok) + C++20 patch |
| **256K `<think>` + Tool Call (`255,693` tok)** | `5,489` slots (`7.36 GiB`) | **`444.9 tok/s`** (`574.2 s` cold, 0% thermal drop) | **`41.5 tok/s`** (`<think>`+JSON) | Emits valid `run_patch_verification` call pulling keys across `243K` tok |

<sub>*Note: Turn-2's `9.63 s` incremental response leverages Strata's built-in session prefix cache (also available in upstream) to avoid re-reading the first `252,163` tokens, demonstrating practical multi-turn Agent responsiveness at `255K` depth.</sub>

---

### 3. Step-by-Step Optimization Breakdown (What Each Layer Delivered)

1. **Step 0 — Host DMA, THP Zero-Copy Pinning & Dual-Fan HBM2 Lock**
   - Unlocked PCIe `MaxReadReq=4096B` + `iommu=pt` (`13.3 GB/s` sustained H2D DMA), pinned the `33.02 GiB` expert arena with `THP [always]`, trimmed idle VRAM footprint to `27 MiB`, and locked MI50 HBM2 at `1000 MHz` (`1,024 GB/s`).
2. **Step 1 & 2 — Custom Vega 20 (`gfx906`) GCN Kernels & Broadwell AVX2 Bitplane Kernels (Zero Flags, 100% Bit-Identical)**
   - **Fused MoE Experts (`src/kernels/cuda/iq_kernels.cu`)**: Enabled single-launch Mode-8 (`Gate+Up+SwiGLU+Q8_1`) across `IQ2_S / IQ2_XXS / IQ1_M`, designed an **11-dword conflict-free LDS bitplane transpose** (`gcd(11,32)=1`) for `Q2_0` down-projection, halved `v_dot4_i32_i8` instructions (`16 → 8`) via branchless two's-complement `nib_neg4`, hoisted multi-token `DecG` lookups, and shrank LDS dynamically (`LNT=1/2/4`, `20.7 KB → 11.3 KB`) to jump from **3 to 5 resident blocks/CU** (300 blocks single-wave across 60 CUs). Real-weight 48-layer MoE sped up **`1.48x` (`T=1`)** and **`1.56x` (`T=3` MTP; `IQ2_XXS` `2.65x`)**, boosting `--serve` exact 4K decode by **`+27.3%`** (`31.5 → 40.1 tok/s`) before touching a single config knob.
   - **Router, BF16/Quant GEMV & 248K Vocab Verify (`native_bf16.cu`, `native_router.cu`, `native_mmvq.cu`, `s26_tsum.cuh`, `verify_kernels.cu`)**: Eliminated 512-CTA wave-tail thrashing on 60 CUs via 128-CTA 4-row tiles (`4.35x–5.67x` faster `[512][2560]` router GEMV; **`3.05x` faster 48-layer routing chain**), replaced 67%–80% of `ds_bpermute_b32` LDS shuffles with 1-cycle hardware VALU DPP (`row_shl` / `quad_perm`), and parallelized Wave64 248K-vocab `argmax_rows` (**`7.84x` faster**).
   - **Broadwell AVX2 4-Row Bitplane Pool (`src/kernels/cpu/q2_avx2.cpp`, `iq_avx2.cpp`)**: Added 4-row `ymm` register-blocked `rows_bp4_nt1` with 12-row software prefetch and `IQ1_M` multi-token AVX2 kernels (`6.23x` vs `ggml` / `2.70x` vs default `q2_avx2` single-core cold; **`1.73x` faster 17-core CPU `down` phase at `27.4 GB/s`**).
3. **Step 3 — CPU–GPU–PCIe Co-Scheduling & Vega 20 Exact `arch_defaults` (`+24.6%` to `+29.1%` Decode)**
   - Microsecond timeline profiling showed the 18-core CPU pool idling `26 ms/round` at default `pcie_frac=0.37`. Rebalancing to **`pcie_frac=0.14`** (86% missed experts on 17 AVX2 cores, 14% over PCIe DMA) + **`spec_min_p=0.65`** + **`aux_cpus=1`** (async hot-expert swap on Core 35 reaching `93%–98%` VRAM hit rate) + baking **17 verified exact `gfx906` switches** into `src/core/arch_defaults.cpp` (while pruning two Vega 20 regressions `FUSE_CONVAB` and `GR_SPLIT` that cost `1.26 ms/window`) lifted exact chat decode to **`46.2 tok/s`** and practical code decode to **`55–57 tok/s`**.
4. **Step 4 — Shipped 112-Shape `gfx906` `rocBLAS` FP16 Table (`+28.1%` 4K Prefill)**
   - Unlocked `tune_rocblas` on `gfx906` and generated `tools/hip/gfx906-rocblas-5.5.0.cd957402309.txt` (100 of 112 GEMM shapes >5% faster; GDN `[10240, 2560]` **`2.05x–2.45x`**). Paired with `STRATA_HIP_PROMPT_F16=1` + `STRATA_HC_UPMIX=1`, hyper-connection `hc down+silu` time dropped from **`26.3%` to `3.4%`**, lifting 4K prefill from `307.1` to **`393.3 tok/s`**.
5. **Step 5 — Hand-Written Wave64 `v_dot4_i32_i8` Online FlashAttention (`523.3 tok/s` @ 32K, `444.9 tok/s` @ 256K)**
   - Implemented `prompt_attn_gfx906_kernel` in `src/kernels/cuda/qsa_prompt_attn.cu` for matrix-core-less Vega 20: `4 × Wave64`, 1-cycle DPP quad-packing/reductions, 24-bit 3-slice `sdot4` dot products (`1.7e-6` max error vs FP64), fused Sigmoid Gate epilogue, and a disjoint-lifetime anonymous `union` in `Smem906` keeping both `int8` (`26.4 KB`) and `k8v4` (`30.1 KB`) **under `32 KB` LDS for 2 blocks / 8 waves per CU** (`2.07x–2.53x` faster kernel). Even at `--max-context 262144 --kv k8v4`, the single 16GB MI50 statically allocates the full 256K KV cache while retaining **`5,489` VRAM expert slots (`7.36 GiB`)** and `667 MiB` free VRAM headroom.

---

### 4. Reproducing on `gfx906` + `Broadwell` (Linux + `rocm-gfx906:7.14`)

```bash
# 1. Build native gfx906 + Broadwell AVX2 engine
cmake -B build-opt -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_HIP=ON \
  -DSTRATA_HIP_GFX906=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_CXX_FLAGS="-O3 -march=broadwell -mtune=broadwell"
cmake --build build-opt -j"$(nproc)"

# 2. Launch 256K Agent Server (or set --max-context 32768 --kv int8 for daily Throughput-Max)
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export STRATA_Q2_BITPLANE=1 STRATA_CPU_IQ_ALL_EXACT1=1
export STRATA_ADAPT_OVERLAP=1 STRATA_ADAPT_JOB_CPU=35 STRATA_SPEC_COST=1
export STRATA_HIP_PROMPT_F16=1 STRATA_HC_UPMIX=1 STRATA_GFX906_PA=1
export STRATA_ROCBLAS_TUNING="$PWD/tools/hip/gfx906-rocblas-5.5.0.cd957402309.txt"

python3 serve/server.py --engine strata --host 127.0.0.1 --port 8090 -- \
  ./build-opt/strata --serve \
  --pack /path/to/packs/iq2_xs \
  --native /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --ple-gguf /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf \
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto \
  --spec 4 --spec-min-p 0.65 --mtp /path/to/mtp/rt \
  --max-context 262144 --kv k8v4 --pcie-frac 0.14 \
  --pool-workers 17 --host-core last --aux-cpus auto --vram-reserve-mib 768 \
  --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

<hr>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Run powerful AI on your own computer.</h3>
<p align="center">Chat, code and work with images, locally.</p>

<p align="center">
  <a href="#models">Models</a> · <a href="#install"><b>Install</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#tools">Tools</a> ·
  <a href="#help">Help</a> · <a href="#support-strata">Support</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Download Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff" alt="Sponsor Strata on GitHub Sponsors"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Support Strata on Buy Me a Coffee"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

**Free, open source, and yours to run.** Use Strata on your desktop or with your favorite coding tools. No account needed.

## Models

We recommend **ISTA-DASLab first**, then **Unsloth** and other providers, where a compatible model fits.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provider | Choices |
| --- | --- |
| **ISTA-DASLab · recommended** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) for coding. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Lower memory use but slower. |
| **Other options** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compare sizes, memory needs and compatibility](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. These are the currently documented packs for this model.

[Model guide and compatibility](docs/QWEN36.md).

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

In the **desktop app** and the **Terminal**. What does not fit in memory is read from your SSD, so it needs a lot of free
disk space.

| Set (ISTA-DAS) | Download | Free disk needed |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | about 245 GB |
| 2.00 bpw | 257 GB | about 265 GB |
| 2.25 bpw | 274 GB | about 280 GB |
| ISTA-DAS Coder (half of the experts removed) | 191 GB | about 197 GB |

Runs on NVIDIA RTX 30, 40 and 50 series (Windows and Linux), AMD Radeon AI PRO R9700 (Linux), AMD Ryzen AI Max — Strix Halo
(Windows), and Apple Silicon Macs with 64 GB or more ([Mac](docs/DEEPSEEK_MAC.md)). Setup downloads the files and the DeepSeek engine, then starts the model.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Measured with the 2.00 bpw set on Strata 0.2.5 (Strix Halo: 0.2.7). Output is the speed of the answer, prompt is how fast Strata reads your input.

| Hardware | Output | Prompt |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB RAM | 4 to 6 tokens/s | about 110 tokens/s (4K) |
| RTX 3060 12 GB, 46 GB RAM | 9 to 11 tokens/s | about 155 tokens/s (4K) |
| 1x Radeon AI PRO R9700, 251 GB RAM | 21 to 25 tokens/s | 320 to 360 tokens/s (8K) |
| 4x Radeon AI PRO R9700 | 74 to 77 tokens/s | 400 to 450 tokens/s (8K) |
| Ryzen AI Max+ 395 (Strix Halo), 128 GB unified | 19 to 21 tokens/s (code), 11 (chat) | 87 tokens/s (4K), 125 tokens/s (8K) |

[Memory, disk and details](docs/DEEPSEEK41.md) · [Release notes](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

Not sure what fits? [Choose a model for your computer](docs/MODELS.md#pick-by-memory).

## Install

<a id="option-a-the-strata-app"></a>

### Desktop app

**[Download Strata for Windows, Linux or Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Install the app, choose a model, and start chatting. There is one installer for each system. On first start the app
downloads the engine for your graphics card. [Help with the installer](docs/INSTALL.md#the-app).

<sub>Files: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata desktop app with a chat and live code preview side by side">
  <br><sub>Chat, files and a live preview in one window.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Terminal app

Prefer the terminal? Install Strata, then run **`strata launch`**.

<details open>
<summary><b>Install commands</b></summary>

**Linux / Mac**

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

On Windows, open a new terminal afterwards. Then:

```sh
strata launch
```

Follow the prompts to download and start a model. [Terminal guide](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata terminal app showing model activity, speed, memory and recent requests">
  <br><sub>Watch your model and hardware while it runs.</sub>
</p>

<a id="build-from-source"></a>

[Build from source](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Full install guide](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

| Supported hardware | Guide |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 series** | [NVIDIA setup](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [AMD setup](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [RX 6000 setup](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel setup](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 or newer** | [Mac setup](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo setup](docs/STRIX_HALO.md) |
| **NVIDIA DGX Spark · Linux** | [DGX Spark setup](docs/DGX_SPARK_LINUX.md) |

<a id="using-it"></a>

## Tools

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, and more through the OpenAI- and Anthropic-compatible APIs.

Run **`strata connect`** to see setup instructions for your tool. [Connection guide](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Help

[All guides](docs) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Report a bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** A GitHub sponsorship or a coffee helps fund development and testing on more hardware.

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: sponsor Strata" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

### Loyal supporters

Thank you to everyone who supports Strata.

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

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a> · <a href="docs/AI_SETUP.md">setup instructions</a>.</sub>

**Sponsorships, hardware partnerships or collaborations:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).

