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
| **CPU** | **Intel Xeon E5-2686 v4** (14nm Broadwell-EP, **18 Cores / 36 Threads**, `3.00 GHz` sustained all-core AVX2 + FMA3 under `performance` governor, Uncore Ring/LLC locked at **`2.70 GHz`** via `MSR 0x620 = 0x1b1b`, `45 MiB` 20-way L3 cache partitioned via `/sys/fs/resctrl` hardware CAT: **`18 ways (40.5 MiB)`** for CPUs `0–17` AVX2 compute / **`2 ways (4.5 MiB)`** isolated for CPUs `18–35` background DMA) |
| **System RAM** | **64 GB (`4 × 16 GB`) Quad-Channel DDR3-1866 ECC Registered** (`59.7 GB/s` theoretical peak; `33.02 GiB` sparse expert arena + `2.39 GiB` 256K QSA KV stream zero-copy pinned via `Transparent Huge Pages [always]` + `cudaHostRegister(PORTABLE)`) |
| **GPU** | **AMD Instinct MI50 16GB HBM2** flashed with **AMD Radeon Pro VII** vBIOS (`Vega 20` / `gfx906:sramecc+:xnack-`, 7nm, **60 CUs / 3,840 SPs**, Wave64, `64 KiB` LDS/CU, **NO Matrix / Tensor Cores**, native `v_dot4_i32_i8` + `26.8 TFLOPS` FP16 + 1-cycle VALU DPP + async Global-to-LDS `buffer_load_dword ... lds` DMA; runtime Linux soft `pp_table` unlocking **`1700 MHz` SCLK / `1100 MHz` HBM2 (`1,126.4 GB/s`) / `1240 MHz` FCLK** without hardware SPI flashing, **`0` UMC/GFX RAS ECC errors**, `15.957 GiB` usable VRAM via AST BMC headless display) |
| **Thermal Mod** | Custom dual-fan shroud mod + `hwmon` PWM daemon (`strata-mi50-guard`): holds **`32–48 °C` edge / `42–60 °C` HBM2 / `47–84 °C` junction** across continuous full-load 256K prefill & Agent workloads (zero thermal throttling, `ue: 0, ce: 0` RAS) |
| **PCIe Interconnect** | **PCIe 3.0 x16** (`iommu=pt` passthrough, `setpci` unlocked **`MaxReadReq = 4096 Bytes`** from BIOS `256B`, ASPM L1 disabled) → **`13.3 GB/s` sustained H2D pinned DMA** |
| **NVMe Storage** | PCIe NVMe SSD streaming the `26.8 GiB` (`320M`-row) Per-Layer Embedding (PLE) n-gram shard via unbuffered `O_DIRECT` at **`38,069 rows/s`** |
| **OS & Compiler** | Ubuntu 24.04 LTS (Linux 6.8, `ulimit -l unlimited`, `vm.swappiness=1`), host-extracted native **`rocm-gfx906:7.14`** (`AMD clang 23.0.0git +PATCHED`, `rocBLAS 5.5.0.cd957402309`, `-O3 -march=broadwell -mtune=broadwell`, `CMAKE_HIP_ARCHITECTURES=gfx906`) |
| **Evaluated Model** | **ISTA-DASLab `Qwen3.8-Flash-Next` (`GSQ-RCO-IQ2_XS`)** (≈125B sparse MoE, 512 experts top-10, 36 GDN linear-recurrence + 12 QSA sparse block-attention layers, `36.5 GiB` Shard 1 + `26.8 GiB` PLE Shard 2 + `952 MiB` MTP speculative drafter, native `262,144` trained context) |

---

### 2. End-to-End Performance Scorecard

#### 2A. Strict Apples-to-Apples E2E Comparison (Frozen Vanilla Upstream vs. Final Full-Stack Optimized Fork)

Every row below uses the **exact same prompts, random seeds, and token budgets**, comparing unmodified upstream Strata (`build-baseline`) against our final pure-software + OS/GPU-kernel optimized build (`build-opt`, zero hardware SPI flashing):

| Workload (`Qwen3.8-Flash-Next` `IQ2_XS` + MTP `--spec 4`) | Vanilla Upstream (`Baseline`)* | Final Optimized Fork (`build-opt`) | Measured E2E Gain | Verification & Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Interactive Chat / Code Decode (9-Run Median / Mean)** | `36.60 tok/s` (`39.77` mean) | **`55.60 tok/s`** (`59.98` mean) | **`+51.9%`** | 9-run suite (**`56.90 tok/s` median even at full `256K` cap**) |
| **Code Generation Decode (Peak — Linear `--spec 4`)** | `48.80 tok/s` | **`71.60 tok/s`** (`76.3–77.8` warm) | **`+46.7%`** | High 4-token MTP acceptance + high-priority MTP HIP stream |
| **Code Generation Decode (Opt-In `--spec-tree 3 --suffix-draft 2`)** | `48.80 tok/s` | **`75.40` / `83.80 tok/s`** | **`+54.5%` / `+71.7%`** | Multi-branch MTP Top-2 + Suffix tree verify (`merge` / warm refactor) |
| **Zero-Skip Lossless Chat (`Exact_Max`, `TAIL_SKIP=0`)** | `35.80 tok/s` (`E0`) | **`55.20 tok/s`** (`56.92` mean, `62.2` peak) | **`+54.2%`** | CLI 4K + MTP 128-token greedy **100% bitwise identical** (`47.21 tok/s`) |
| **Zero-Skip Lossless 4K Decode (`TAIL_SKIP=0`)** | `31.50 tok/s` (`E0`) | **`47.80` / `48.60 tok/s`** | **`+51.7%` (cold)** | Pure lossless GCN LDS-DMA + VALU DPP + AVX2 + OS stack gain |
| **4K Context Prompt Prefill (Cold / Warm Profile)** | `308.30 tok/s` | **`550.70` / `614.00 tok/s`** | **`+78.6%` / `+99.2%`** | Nearly **`2.0x`** faster warm prefill (`I=128` MMQ + `rocBLAS` + `gfx906` PA) |
| **4K Context Long Decode (Cold / Warm Profile)** | `35.50 tok/s` | **`54.20` / `49.80 tok/s`** | **`+52.7%` (cold)** | `resctrl` L3 CAT eliminates DMA thrashing (`16x` lower latency jitter) |
| **32K Long-Context Prefill (`27,818` tok, `int8` KV)** | `345.70 tok/s` measured (`438.10` w/ HW tuning)† | **`685.70 tok/s` (`40.6 s`)** | **`+98.4%`** (**`+56.5%`** pure SW) | Nearly **`2.0x`** faster, saves **`~40 s`** per prompt; 100% needle recall |
| **32K Non-Copy Long Decode (`27,818` tok, `256` tok out)** | `≈38.6 tok/s` est. (`49.20` w/ HW tuning)† | **`55.00 tok/s`** | **`≈+42.5%`** (**`+11.8%`** pure SW) | 256-token original OoO CPU pipeline essay (zero archive copy inflation) |
| **64K Agent Prefill (`64,452` tok @ Full `256K` Cap)** | `≈331.7 tok/s` est. (`420.40` w/ HW tuning)† | **`655.70 tok/s` (`98.3 s`)** | **`≈+97.7%`** (**`+56.0%`** pure SW) | Saves **`~96 s`** vs stock HW (`55.0 s` pure SW); 100% needle recall |
| **64K Agent Non-Copy Decode (`64,452` tok @ Full `256K` Cap)** | `≈33.3 tok/s` est. (`42.50` w/ HW tuning)† | **`54.60 tok/s`** | **`≈+64.0%`** (**`+28.5%`** pure SW) | Vanilla drops to `5,539` slots; **ours holds `7,499` slots (`55.0→54.6`, 0% drop)** |
| **248,320 Full-Vocab FP64 Forward KL Divergence** | `0.009351` | **`0.006085` (`5/5` Top-1 match)** | **`-34.9%` KL error** | `Exact_Max` KL `0.002010`; SpecPilot `5,760`-tok Top-1 **`99.27%–99.38%`** |

<sub>*Note 1 (Rows 1–7 frozen baseline): Rows 1–7 were recorded at stock GPU clocks (`1000 MHz` HBM2 / `1153 MHz` SCLK) with `--max-context 32768` only, atop Step-0 PCIe `4096B` (`13.3 GB/s` DMA) + `THP [always]` host pinning.<br>†Note 2 (32K & 256K-cap 64K non-copy head-to-head + calibrated stock-hardware baseline): To eliminate verbatim prompt-copy Suffix Draft inflation, both 32K (`27,818` tok) and 256K-cap 64K (`64,452` tok) were re-measured back-to-back using **passcode retrieval + a `256`-token original essay on out-of-order CPU pipelines**. Parenthetical `w/ HW tuning` values were measured directly on unmodified `build-baseline` while retaining our host/GPU hardware tuning (`1700/1100/1240 MHz` soft `pp_table` + `2.70 GHz` Uncore + `resctrl` L3 CAT); `est.` values divide out our Stage-A measured hardware clock multipliers on this exact card (`438.10 / 345.70 = 1.267x` prefill, `52.8 / 41.4 = 1.275x` decode; reverting BIOS `MaxReadReq=256B` & `THP [madvise]` lowers vanilla 256K decode another `~5%–8%` to `~30–31 tok/s`). Crucially, **enabling `--max-context 262144` slashes vanilla resident VRAM experts from `7,518` to `5,539` slots (`7.43 GiB`), dropping decode from `49.2 → 42.5 tok/s` (`≈33.3 tok/s` stock HW), whereas our `--kv-resident 16384` + `mtp.cpp` fix preserves `7,499` slots (`10.04 GiB`) and holds `54.60 tok/s` at 64.5K depth — virtually zero drop from 32K (`55.00 tok/s`)!**</sub>

#### 2B. 64K → 256K Extreme Context Scaling & Multi-Turn Agent Stress Test (Single 16GB MI50)

Measured on the optimized fork (`--kv k8v4`) using non-repetitive multi-module C++20 repository snapshots and long-horizon archive logs on a single 16GB HBM2 card (`—` where vanilla upstream was not run at 128K–256K):

| Agent Stage & Context Depth | Vanilla Upstream (`256K` Cap) | Resident VRAM Experts (`build-opt`) | Prefill / Incremental TTFT | Decode Speed | Accuracy / Agent Workflow Verification |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **64K Agent Ingestion (`64,452` tok, `--kv-resident 16384` @ 256K cap)** | `5,539` slots (`420.4 / 42.5 tok/s` w/ HW; `≈331.7 / ≈33.3` stock HW) | **`7,499` slots (`10.04 GiB`)** | **`655.7 tok/s`** (`98.3 s` cold) | **`54.6 tok/s`** (`256`-tok non-copy essay) | 100% passcode recall; `92.4%` QSA KV block reads hit VRAM |
| **128K Full Cold Repo (`127,949` tok, static KV)** | `—` | `5,489` slots (`7.36 GiB`)* | **`484.3 tok/s`** (`264.2 s` cold) | **`38.9 tok/s`** (pure MTP) | **2/2** audit keys (`19.1K` & `108.2K`) recalled verbatim |
| **256K Extreme Cold Turn 1 (`251,964` tok, static KV)** | `—` | `5,489` slots (`7.36 GiB`)* | **`444.9 tok/s`** (`566.4 s` cold) | **`35.6 tok/s`** (pure MTP) | **3/3** keys (`12.4K / 125.4K / 238.7K`) recalled verbatim |
| **256K Prefix-Cached Turn 2 (`255,042` tok, static KV)** | `—` | `5,489` slots (`7.36 GiB`)* | **TTFT `9.63 s`** (`271.8 tok/s` on `2,619` new tok)** | **`54.8 tok/s`** (Suffix+MTP) | Reuses Strata's native prefix cache (`252,163` tok) + C++20 patch |
| **256K `<think>` + Tool Call (`255,693` tok, static KV)** | `—` | `5,489` slots (`7.36 GiB`)* | **`444.9 tok/s`** (`574.2 s` cold, 0% thermal drop) | **`41.5 tok/s`** (`<think>`+JSON) | Emits valid `run_patch_verification` call pulling keys across `243K` tok |

<sub>*Note 1: The `128K–256K` cold repository stress runs above were recorded prior to Step 6 (`--kv-resident 16384` + `mtp.cpp` dense-ring fix), which subsequently expanded full `262,144`-context resident VRAM capacity from `5,489` (`7.36 GiB`) to **`7,499` slots (`10.04 GiB`)**.<br>**Note 2: Turn-2's `9.63 s` incremental response leverages Strata's built-in session prefix cache (also available in upstream) to avoid re-reading the first `252,163` tokens, demonstrating practical multi-turn Agent responsiveness at `255K` depth.</sub>

---

### 3. Step-by-Step Optimization Breakdown (What Each Layer Delivered)

1. **Step 0 — Host DMA, Uncore `2.70 GHz`, Intel RDT `resctrl` L3 CAT & Soft `pp_table` (`1,126 GB/s` HBM2, Zero Hardware Flash)**
   - Unlocked PCIe `MaxReadReq=4096B` + `iommu=pt` (`13.3 GB/s` sustained H2D DMA), pinned the `33.02 GiB` expert arena with `THP [always]`, locked Broadwell Uncore Ring/LLC at `2.70 GHz` (`MSR 0x620 = 0x1b1b`), partitioned the `45 MiB` 20-way L3 cache via `/sys/fs/resctrl` hardware CAT (`18 ways` for CPUs `0–17` AVX2 workers / `2 ways` for CPUs `18–35` background DMA, cutting decode latency std-dev **16x** from `1.79` to `0.11 tok/s`), and injected a runtime Linux soft `pp_table` unlocking **`1700 MHz` SCLK / `1100 MHz` HBM2 (`1,126.4 GB/s`) / `1240 MHz` FCLK** under `strata-mi50-guard` (`ue: 0, ce: 0` RAS ECC).
2. **Step 1 & 2 — Custom Vega 20 (`gfx906`) Zero-Spill Wave64 Kernels, `I=128` MMQ & Broadwell Dual/4-Row AVX2 Pool**
   - **Fused MoE & Prefill MMQ (`iq_kernels.cu`, `mmq_moe.cu`)**: Enabled single-launch Mode-8 (`Gate+Up+SwiGLU+Q8_1`), designed an **11-dword conflict-free LDS bitplane transpose** (`gcd(11,32)=1`) for `Q2_0` down-projection, halved `v_dot4_i32_i8` instructions (`16 → 8`) via branchless `nib_neg4`, shrank LDS dynamically (`20.7 KB → 11.3 KB`) for **5 resident blocks/CU** with **0 VGPR spills** across all `gfx906` translation units, and added `I=128` 2-pass sorted MoE MMQ tiles (`STRATA_MMQ_GROUP_SORT=1`).
   - **Router & 248K Vocab Verify (`native_router.cu`, `verify_kernels.cu`)**: Eliminated 512-CTA wave-tail thrashing on 60 CUs via 128-CTA 4-row tiles (**`3.05x` faster 48-layer routing chain**) and parallelized Wave64 248K-vocab `argmax_rows` (**`7.84x` faster**).
   - **Broadwell AVX2 Bitplane & Dual-Row Pool (`q2_avx2.cpp`, `iq_avx2.cpp`)**: Added 4-row `ymm`-blocked `rows_bp4_nt1` (`6.23x` vs `ggml` cold; **`1.73x` faster 17-core CPU `down` phase**) plus dual-row `IQ2_XS` gate/up sharing L1 `q8_1` activation loads across expert pairs (`+6–8%` CPU pool throughput).
3. **Step 3 — Bare-Metal GCN Inline Assembly (`buffer_load_dword ... lds` Async DMA & Engine-Wide 1-Cycle VALU DPP)**
   - Bypassed compiler VGPR staging in `iq_kernels.cu` and `native_q2.cu` via hand-built 128-bit GCN Buffer Resource Descriptors (`SRD` in `s_mov_b32`, `S_DATA_FORMAT=4` hardware bounds clamp) + `m0` LDS routing + **`buffer_load_dword v0, v_off, s_srd, 0 offen offset:0 lds`** (`308` hardware Global-to-LDS DMA instructions streaming `q8_1` activations straight from HBM2 into LDS, paired with `s_waitcnt lgkmcnt(0); s_barrier` so in-flight `vmcnt` weight prefetches never stall). Replaced remaining `ds_bpermute_b32` cross-lane shuffles across 7 HIP/CUDA headers (`verify_kernels.cu`, `native_bf16.cu`, `native_quant.cu`, `gdn_umv_tc.cuh`, `qsa_kernels.cu`, `mtp_kernels.cu`) with 1-cycle VALU DPP (`v_add_f32_dpp` / `v_mov_b32_dpp`) + `ds_swizzle_b32` (`+6.2%` chat & `+10.6%` 4K decode on top of Steps 0–2, **100% bitwise greedy parity**).
4. **Step 4 — CPU–GPU–PCIe Co-Scheduling (`pcie_frac=0.18`) & High-Priority MTP Hardware Stream (`STRATA_MTP_HIP_PRIO=1`)**
   - Rebalanced missed experts (`82%` on 17 AVX2 cores, `18%` over PCIe DMA) + `spec_min_p=0.65` + async Core-35 hot-expert swaps (`93%–98%` active VRAM hit rate) + 17 baked exact `gfx906` switches in `arch_defaults.cpp`, and isolated MTP speculative drafting onto `cudaStreamCreateWithPriority(..., leastPriority - 1)` (`STRATA_MTP_HIP_PRIO=1`) to preempt background PCIe expert swaps (cutting MTP draft latency **40%**, `5.24 → 2.95–3.35 ms/window`).
5. **Step 5 — Shipped 112-Shape `gfx906` `rocBLAS` FP16 Table & Wave64 `v_dot4_i32_i8` Online FlashAttention (`666 tok/s` @ 32K)**
   - Shipped `tools/hip/gfx906-rocblas-5.5.0.cd957402309.txt` (`2.05x–2.45x` GDN projections) + `STRATA_HIP_PROMPT_F16=1` + `STRATA_HC_UPMIX=1` (`hc down+silu` `26.3% → 3.4%`), and hand-wrote `prompt_attn_gfx906_kernel` in `qsa_prompt_attn.cu` (`4 × Wave64`, 24-bit 3-slice `sdot4` with `1.7e-6` max error vs FP64, fused Sigmoid Gate epilogue, and `<32 KB` disjoint-lifetime `Smem906` union achieving **2 blocks / 8 waves per CU**), driving 4K warm prefill to **`614.0 tok/s`** and 32K–64K prefill to **`656.7–666.0 tok/s`**.
6. **Step 6 — 256K Pinned-RAM KV Streaming (`--kv-resident 16384`) & MTP Dense Ring VRAM Fix (`+2,010` Expert Slots at 256K)**
   - Fixed `src/model/mtp.cpp` so MTP dense ring allocation honors `kv_resident_cells` (`-263 MiB` MTP VRAM: `1,100 → 837 MiB`) and streamed cold 256K QSA `k8v4` KV blocks from `2.39 GiB` pinned DDR3 RAM while keeping `16,384` hot cells/layer in HBM2 (`92.4%` block-attention VRAM hit rate at 64K). Full `262,144`-context resident VRAM experts jumped from `5,489` (`7.36 GiB`) to **`7,499` slots (`10.04 GiB`)**, lifting 54K–64K Agent decode from `~30` to **`48.2–51.1 tok/s`** without lowering `max_context=262144`.
7. **Step 7 — Opt-In Multi-Branch Tree Speculative Verification (`--spec-tree 1/2/3`, Up to `83.8 tok/s` Code Decode)**
   - Added end-to-end 2-branch speculative tree drafting (device-side MTP Step-0 Top-2 + Suffix divergence) and zero-spill 48-layer causal-tree verification across GDN delta-rule recurrence (`tree_parent[t]` state branching), QSA sparse block-attention (`tree_anc_mask[t]` + Clock-LRU resident compaction), and PLE n-gram hashing. Verified across `5,760` continuation tokens via SpecPilot FP64 logit forcing (`99.38%` Top-1 match on `--spec-tree 3` vs `99.27%` stock `--spec 4`), reaching **`75.4–83.8 tok/s`** on structured code generation.

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

# 2. Launch 256K Agent Server (for 32K daily mode, use --max-context 32768 --kv int8 --kv-grow --vram-reserve-mib 704;
#    for structured code workloads, optionally append --spec-tree 3 --suffix-draft 2)
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

