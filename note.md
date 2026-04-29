# Strategies for cuFFT-like startup behavior with rocFFT

rocFFT compiles FFT kernels at runtime via the `rocfft_rtc_helper` subprocess
on first use, while cuFFT ships pre-built kernels in its library. The
difference is most visible as multi-second first-run latency the first time a
new (shape, precision, batch, stride, in/out-place) combination is used.

A useful baseline observation on a mixed-install box: AMD's official rocFFT 6.4
package shipped a pre-populated kernel cache DB
(`/opt/rocm-6.4.3/lib/rocfft/rocfft_kernel_cache.db`), while the Ubuntu archive
rocFFT 7.1 package did not. That same mechanism — a pre-warmed kernel DB next
to the library — is what gets you cuFFT-like startup.

## Strategy, layered from easiest to most thorough

### 1. Reuse plan handles (foundation)

Single biggest lever. Create plans once, reuse forever. No strategy below
helps if plan handles are recreated in a hot loop.

### 2. Use a *system* kernel cache (read-only, shared)

rocFFT consults a system cache before the user cache. AMD's shipped binary
already populates one — that's what `rocfft_kernel_cache.db` next to the
library is.

- Point a build/runtime at one with
  `ROCFFT_SYS_KERNEL_CACHE_PATH=/path/to/rocfft_kernel_cache.db`.
- Or drop a pre-warmed `.db` into the rocFFT install dir so all users on the
  host benefit without env vars.

This is the closest thing to "ship kernels in the library" that rocFFT
provides.

### 3. Pre-warm at build/install/CI time, then ship the cache

The actual "make rocFFT feel like cuFFT" move:

1. Write a tiny warmup binary that creates a `rocfft_plan` (or `hipfftHandle`)
   for every (shape, precision, batch, in/out-place, stride) combo the app
   actually uses.
2. Run it once on a build machine with the **same gfx target** (e.g. gfx1200).
3. Capture the resulting `~/.cache/rocFFT/rocfft_kernel_cache.db`.
4. Ship that file with the app — install it as the system cache, copy it into
   `~/.cache/rocFFT/` on first launch, or bake it into the Docker image.

For Docker: add a `RUN /opt/yourapp/warmup` step in the image build so the
cache is a layer in the final image. End users see cuFFT-like startup.

### 4. Canonicalize shapes in application code

Every distinct (length, batch, stride, precision, R2C/C2C/C2R) is a separate
kernel set. Code that does FFTs on whatever shape the input happens to be
will keep missing the cache. Pad/crop to a small set of canonical sizes
(power-of-2 lengths, fixed batch chunks). This makes both the warmup feasible
and the cache hit rate ~1.

### 5. AOT-compile kernels into a library (most ambitious)

rocFFT's source build supports `rocfft_aot_helper` to ahead-of-time compile a
chosen set of kernels and bake them into the library, eliminating RTC for
those shapes entirely. Build rocFFT from source with a manifest of kernels to
pre-build. Heavy effort — only worth it for shipping a product where FFT
shapes are fully controlled and zero-RTC behavior is needed even with no user
cache.

### 6. Minor: compile in-process

`ROCFFT_RTC_PROCESS=0` runs compilation in-process instead of spawning
`rocfft_rtc_helper`. Saves the subprocess fork overhead but does **not** avoid
compilation. Useful in environments where forking is restricted (some
sandboxes, certain MPI setups). Don't expect dramatic gains.

## Pragmatic path on this box

This box is on Ubuntu 26.04 with rocFFT 7.1.1 from the Ubuntu archive (no
pre-shipped kernel DB) on a gfx1200 (RX 9060 XT).

1. Build a one-shot warmup tool listing real FFT shapes used by finufft.
2. Run it on the 7.1.x stack, on this gfx1200 box.
3. Copy the resulting `~/.cache/rocFFT/*.db` into
   `/usr/lib/x86_64-linux-gnu/rocfft/` (or similar) and point
   `ROCFFT_SYS_KERNEL_CACHE_PATH` at it for shared use.
4. First-run latency for those shapes is gone permanently.

## Relevant env vars

- `ROCFFT_RTC_CACHE_PATH` — relocate the user-writable kernel cache (handy
  for shared/read-only homedirs or CI).
- `ROCFFT_SYS_KERNEL_CACHE_PATH` — read-only system cache consulted first.
- `ROCFFT_RTC_PROCESS=0` — compile in-process instead of spawning a helper.
- `ROCFFT_DEBUG_GENERATOR=1` — dump generated HIP source for inspection.

# finufft use

## Why finufft hits rocFFT harder than typical users

- **Shapes are user-driven, unbounded.** Users hand finufft
  `(M1, M2, M3, ntrans, upsampfac, prec)` whatever their science demands. The
  padded grid size `nf_i` depends on all of those, and `next235` /
  `next235beven` produce many distinct sizes. The kernel cache effectively
  has no working set across the user base — every new problem is a cold
  cache.
- **One-shot API patterns are the common case.** MATLAB/Python users mostly
  call `finufft1d1` / `finufft2d2` / etc., which internally create plan → do
  work → destroy plan. On cuFFT that's milliseconds; on rocFFT each unique
  padded shape pays the full compile bill *every cold-cache process*.
- **Type 3 grid sizes are even worse.** Type 3 picks `nf` from a non-uniform
  spread bandwidth × point geometry. The shapes are essentially never round
  numbers and rarely repeat across calls.
- **Multiplicity stacks fast.** {1D, 2D, 3D} × {type 1, 2, 3} × {single,
  double} × {upsampfac=2.0, 1.25} × {ntrans variants} × {dozens of `nf_i` per
  dim} → the kernel set finufft can demand from rocFFT is huge.
- **Benchmarking gets misread.** First-run timings dominated by compile look
  like finufft is slow on AMD when it's actually rocFFT bootstrapping. This
  is a reputational tax on the library, not just a UX tax.
- **CI fresh containers.** Every CI run with no cache pays full price.

## What finufft, as a library, can actually do about it

Ranked by effort × benefit:

1. **Documentation + a startup-time warning.** When ROCm is detected and the
   cache is empty/non-writable, emit a one-shot stderr note explaining what's
   about to happen. Cheap, immediately reduces the "is finufft broken?"
   support burden.

2. **Strongly steer users to the guru / plan API on AMD.** The
   `finufft_makeplan` / `setpts` / `execute` flow keeps the rocFFT plan
   handle alive, so the compile cost is paid once per process. Already
   exists — just needs prominent ROCm-specific docs and example code paths.

3. **Coarsen `next235` on ROCm.** Biggest lever inside finufft itself.
   Instead of finding the smallest 2/3/5-smooth `nf ≥ ceil(upsampfac·M)`,
   snap up to a coarser quantization (e.g. multiples of 32, or only powers
   of 2, or a hand-picked ladder). Spend ~10–30% more FFT work per call but
   cut the distinct-shape set by orders of magnitude, so the cache becomes
   finite and warmup actually works. Gate behind a flag like
   `FINUFFT_FFT_BACKEND_ROCFFT_COARSEN_GRID` with a tunable.

4. **Ship a `finufft-warmup-rocfft` utility.** Takes user-specified ranges
   of `(M1..M3, ntrans, prec, upsampfac, type)` and creates+destroys plans
   for the resulting `nf` set, populating `~/.cache/rocFFT/`. Document
   running it once after install. Pairs naturally with #3 because the
   warmup space becomes finite.

5. **CMake option to pre-warm at build time.** `-DFINUFFT_ROCFFT_PREWARM=ON`
   runs the warmup over a default shape set during `cmake --build` if a gfx
   target matching the build host is present. Build packagers (conda-forge
   etc.) get a populated DB they can ship.

6. **Backend abstraction → consider VkFFT as an AMD path.** Real work, but
   VkFFT has been the workaround in several of the gfx1200 issues, and it's
   been integrated as an FFT backend in other libraries. Worth at least
   scoping whether a `FINUFFT_FFT_BACKEND=vkfft` build path is feasible — it
   would sidestep the entire rocFFT RTC story (VkFFT does runtime kernel
   generation too, but its compile model is reportedly faster and its
   consumer-Radeon support is more mature).

7. **(Long shot) AOT-compile rocFFT with finufft's shape distribution.**
   Build rocFFT from source with `rocfft_aot_helper` over a manifest of
   finufft-typical shapes, link against that custom rocFFT. Highest
   engineering cost, lowest user friction at runtime. Probably only worth it
   for a curated "finufft on AMD" distribution.

## Recommended first ship

If picking one thing first: **#3 (coarsen `next235` on ROCm) + #4 (warmup
utility)** as a paired feature, behind a single `FINUFFT_AMD_FAST_STARTUP`
flag. That combination converts an unbounded compile problem into a bounded
one and gives users a single command to make it disappear. #1 and #2 are
essentially free and should ship anyway.

# Malloc issues

## Summary

`hipMallocAsync` (the stream-ordered allocator, AMD's analog of
`cudaMallocAsync`) is **unreliable on gfx1200 / gfx1201 (RDNA4)** with ROCm
7.1.x. This is a known weak spot — multiple open issues against ROCm,
llama.cpp, vLLM, and Ollama trace failures back to the HIP runtime's
stream-ordered memory / VMM-pool path on RDNA4.

## What the device claims vs. what works

On the RX 9060 XT (gfx1200) on this box:

```c
int has_pools = 0;
hipDeviceGetAttribute(&has_pools, hipDeviceAttributeMemoryPoolsSupported, 0);
// returns 1
```

The runtime **advertises** memory-pool support, but the implementation has
bugs. This is the worst-of-both-worlds case:

- Trivial tests (small alloc, immediate free, single stream) **pass**, so
  smoke tests mislead.
- Failures emerge at scale: many allocations, multiple streams,
  long-running processes, or at process teardown.
- Common failure shapes seen in the wild: leaked memory across streams,
  segfault on `hipFreeAsync` after stream destroy, hang on
  `hipDeviceReset` / process exit (the "GPU stuck at 100%" pattern).

## Why this is plausible / known-broken

The stream-ordered allocator sits on top of HIP virtual memory management
(VMM). On RDNA4 the VMM-backed pool path is the documented weak link:
projects working around it on RDNA4 have to **explicitly** enable VMM-pool
builds (e.g. `-DGGML_USE_VMM=ON` in llama.cpp) or fall back to plain
`hipMalloc`, and even then see OOMs that the Vulkan path handles fine.
There are also HSA-runtime-level bugs on gfx1200 with ROCm 7.1.x —
discovery hangs, queue teardown leaving the GPU pegged at 100%. Anything
exercising stream + memory-pool lifecycle is in that blast radius.

Compounding factors here: gfx1200 is brand-new RDNA4 silicon (AMD's tier-1
testing is on CDNA MI300/325), `hipcc` is `7.1.52801-9999` from the Ubuntu
archive, and the distro (Ubuntu 26.04) is not an AMD-supported ROCm target.

## Mitigations, in order of preference

### 1. Explicit private pool, no release-to-OS

Most likely to dodge the bug — bypasses the default pool, never returns
memory to the OS, avoiding the buggy unmap/teardown paths:

```c
hipMemPool_t pool;
hipMemPoolProps props = {};
props.allocType     = hipMemAllocationTypePinned;
props.handleTypes   = hipMemHandleTypeNone;
props.location.type = hipMemLocationTypeDevice;
props.location.id   = 0;
hipMemPoolCreate(&pool, &props);

uint64_t threshold = UINT64_MAX;
hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold);

void* p;
hipMallocFromPoolAsync(&p, n, pool, stream);
// ...
hipFreeAsync(p, stream);
```

Single long-lived pool, destroyed once at shutdown after
`hipDeviceSynchronize`.

### 2. Drop stream-ordered allocation entirely

If (1) still fails, replace `hipMallocAsync` / `hipFreeAsync` with plain
`hipMalloc` + a size-bucketed reuse cache maintained in user code. Loses
the stream-ordering nicety (caller becomes responsible for not freeing
memory the GPU is still touching), gains rock-solid behavior. Most
production HIP code on consumer Radeon does this anyway.

### 3. Confirm by elimination

Run the same code on a CDNA box (MI100 / MI200 / MI300, or a rented cloud
instance). If it works there, the bug is RDNA4-specific and a focused
issue can be filed against `ROCm/ROCm` with `hipcc --version`, kernel
driver version (`dkms status` or `modinfo amdgpu`), and a minimal repro.

## Reference issues

- ROCm/ROCm#5812 — RX 9070 XT (gfx1200) HSA discovery hang on ROCm 7.1.1.
- ROCm/ROCm#5706 — HIP backend leaves RDNA4 GPU at 100% (HSA teardown).
- lemonade-sdk/llamacpp-rocm#87 — VMM-backed pool needed to fix OOM on
  RDNA3.5/4.
- ggml-org/llama.cpp#21376 — RX 9060 XT (gfx1200) ROCm OOM where Vulkan
  succeeds.
- vllm-project/vllm#40081 — vLLM init failures on gfx1201 (RDNA4).

# Upstream context

State of HIP / ROCm / AMD support in `flatironinstitute/finufft` as of
2026-04-28.

## The one substantive thread

**Discussion #350: "Extending to non-NVIDIA GPU frameworks"** — opened by
ahbarnett on **2023-09-18**, 11 replies, exploratory, no commitments. Key
points:

- ahbarnett opened it citing user requests for non-NVIDIA GPU support.
- **blackwer** (maintainer) on HIP/ROCm: *"We can get AMD support almost
  trivially since it is mostly cuda api compatible."* But flagged that the
  real cost is **porting the spreader/interpolator**, not the FFT call —
  *"the FFT call is a nearly negligible fraction of the actual work."*
- Preferred long-term direction is **SYCL** (single source for AMD / Intel
  / NVIDIA), with **heFFTe** suggested as a portable FFT replacement for
  cuFFT / rocFFT.
- blackwer mentioned a **previously-unmerged HIP PR in the old cufinufft
  repo** (pre-merge into finufft) as a reference starting point.
- Quiet since. No timeline, no concrete work.

## Issue / PR tracker status

- **Zero** issues or PRs mentioning "ROCm".
- **Zero** mentioning "Radeon", "hipFFT", or "rocFFT".
- "HIP" search also returns nothing.

## Relevant CUDA-side issues that portend the AMD port

These already happened on the cuFFT/CUDA side and predict what will recur
on rocFFT/HIP:

- **#445** *"check for availability of `cudaMallocAsync`"* (closed) — they
  already debated whether stream-ordered alloc should be required on
  CUDA. Same debate on AMD has a sharper edge: pool support is advertised
  but broken on RDNA4 (see *Malloc issues* above), so the answer must be
  "do not depend on `hipMallocAsync` for correctness".
- **#417** *"Cufinufft, crash when not on default stream"* (closed) —
  non-default-stream fragility already surfaced on CUDA. Expect similar
  on HIP, especially given the RDNA4 HSA queue / teardown bugs.
- **#818** *"Proposal: Cuda reorganization"* (closed, Feb 2026) and
  **#826** *"Unify and simplify Cuda spreading and interpolation"*
  (merged March 2026) by mreineck — substantial CUDA refactor just
  landed. A HIP port should pattern on this cleaned-up structure rather
  than the older code.

## Implication

There is clear maintainer **interest** in non-NVIDIA, explicit
acknowledgment that the FFT layer is "almost trivial" via HIP, but **no
upstream HIP/ROCm work** in tracker form. If a Q-internal HIP port lands
stable, contributing it back — or even just resurrecting #350 with a
concrete update — would likely be welcomed. The recent CUDA refactor
(#818 / #826) makes *now* a good time to pattern a HIP port on the
post-refactor CUDA structure.

# Metal-accelerated port

Targeting Apple Silicon (M1/M2/M3/M4) for finufft. Bigger lift than HIP
because Metal isn't CUDA-compatible — different shading language (MSL),
different host API, different memory model — but the unified-memory
architecture genuinely simplifies parts of the design.

## What needs porting

Same conceptual decomposition as cuFINUFFT:

1. **Spreader / interpolator kernels** — the dominant cost. Currently
   CUDA C++ kernels with atomic adds (or sort-then-spread). Need MSL
   compute shaders.
2. **FFT call** — currently cuFFT. Need a Metal-side FFT.
3. **Plan / handle / lifecycle** — currently CUDA streams + plan
   bookkeeping. Need `MTLDevice` / `MTLCommandQueue` /
   `MTLCommandBuffer` plumbing.
4. **Host integration** — Objective-C++ or Swift typically; **metal-cpp**
   gives a pure-C++ binding suitable for a library like finufft.

## FFT backend options

- **MPS / MPSGraph** (Apple's first-party): has FFT operations
  (`MPSGraph` `realToHermitean`/`hermiteanToReal` and complex-to-complex
  in newer macOS). First-party, well-supported. Caveat: historically
  restricted to power-of-2 lengths in some paths — must verify whether
  arbitrary 2/3/5-smooth sizes are accepted in current macOS, because
  finufft's `next235` produces 2/3/5-smooth `nf`. If MPSGraph FFT
  rejects these, we'd have to pad to power-of-2 (similar trade-off to
  the rocFFT shape-coarsening discussion above).
- **VkFFT (Metal backend)**: VkFFT has a Metal backend via its
  shader-portability layer. Mature, supports arbitrary 2/3/5/7-smooth
  sizes natively, well-tuned. A clear leading choice. Has the bonus of
  being a candidate for the AMD path too — one FFT abstraction, two
  GPU backends.
- **MLX FFT**: Apple's MLX library has `mlx.fft` backed by Metal.
  Convenient if MLX is acceptable as a heavyweight dep. Probably
  excessive for a C++ library — drag in MLX runtime for one operation.
- **Roll our own** in MSL: only justified if all of the above fall
  short for finufft's specific size set. High effort, no upside vs
  VkFFT.

**Recommendation:** **VkFFT** if we're willing to take the dep and it
clears finufft's correctness/perf bar; **MPSGraph** if we want
first-party-only and can live with size restrictions or coarsening.

## Spreader / interpolator porting

- Translate the CUDA kernels to **MSL compute shaders** (compiled with
  `xcrun -sdk macosx metal` to `.air`, then `metallib`).
- SIMD-group width on Apple Silicon is **32**, matching CUDA warps —
  warp-level idioms (shuffles, ballots) translate via Metal's
  `simd_*` intrinsics.
- Threadgroup memory ≈ CUDA shared memory; sizes per device available
  via `MTLDevice` queries.
- **Atomic float adds** for the scatter path: Metal 3 on Apple Silicon
  exposes `atomic_float` with `atomic_fetch_add_explicit`. Good enough
  for the same `atomicAdd` pattern cuFINUFFT uses. Sort-then-spread is
  also straightforward (Metal has Metal Performance Shaders sort, and
  hand-written radix sort is well-trodden territory).
- **fp64 is a problem.** Apple Silicon GPUs have **no native fp64**;
  MSL emulates it at very low throughput. finufft's double-precision
  default path will be effectively unusable on Metal at competitive
  speed. Options: (a) ship Metal as fp32-only and CPU-fallback for
  fp64 (simple, honest); (b) double-double / compensated fp32 in MSL
  (research-grade); (c) document and accept the slowdown for users who
  need bit-compatible fp64. (a) is the right answer initially.

## Memory model — the unified-memory upside

This is where Metal genuinely beats CUDA/HIP for finufft:

- Apple Silicon GPU and CPU share the same physical memory.
  `MTLBuffer` with `MTLResourceStorageModeShared` is host-readable and
  device-readable with **no explicit copy**.
- The CUDA-side cost of pushing NU points to the GPU on every
  `setpts` largely **goes away**. setpts becomes a pointer hand-off
  plus any required reorganization (sort indices, etc.).
- This changes the algorithmic balance: the relative cost of host-side
  setpts vs GPU-side spread shifts. Worth re-profiling after the port
  rather than assuming cuFINUFFT's hot-path map applies.
- Tradeoff: there's no separate VRAM, so very large problems compete
  with the rest of system RAM.

## Plan / queue / lifecycle

- One `MTLDevice`, one (or a small pool of) `MTLCommandQueue` per
  finufft plan handle.
- Each `finufft_execute` builds a `MTLCommandBuffer`, encodes spread
  kernel → FFT → interp kernel into compute encoders, commits.
  `waitUntilCompleted` for the synchronous API; `addCompletedHandler`
  for async.
- Pipeline state objects (`MTLComputePipelineState`) are the Metal
  analog of compiled CUDA kernels. Build them once at plan creation,
  reuse for every execute. **No equivalent of the rocFFT RTC pain** —
  Metal shader compilation is fast and cached by the system.

## Build system

- Add `FINUFFT_USE_METAL=ON` CMake option, gated on macOS + Apple
  Silicon detection.
- Custom CMake rule: `.metal` → `.air` (`xcrun -sdk macosx metal`) →
  `.metallib` (`xcrun -sdk macosx metallib`); embed the resulting
  `.metallib` either as a sibling file or via a binary-include macro.
- Host side compiled as Objective-C++ (`.mm`) or pure C++ with
  `metal-cpp`. **metal-cpp** is preferable for a library — no ObjC
  runtime requirement for downstream consumers, single-header(ish),
  Apple-blessed.
- Link `Metal.framework` (and `MetalPerformanceShaders.framework` if
  using MPS for FFT, or VkFFT's static lib if going that route).
- CI: add a macOS-arm64 runner; GitHub-hosted `macos-14` / `macos-15`
  is fine.

## Staged plan

1. **Spike**: prove out the path end-to-end with a 1D type-1 fp32
   transform using MPSGraph FFT and a naive atomic-add MSL spreader.
   Compare numerics against CPU finufft. Target: a few hundred lines.
2. **Backend abstraction**: factor cuFINUFFT's plan struct so the GPU
   backend (CUDA/HIP/Metal) is selectable. Likely done well enough
   already after the #818/#826 CUDA refactor — that's the structure to
   pattern on.
3. **VkFFT swap**: replace MPSGraph with VkFFT's Metal backend; verify
   2/3/5-smooth size correctness; benchmark.
4. **Sort-then-spread path**: port the CUDA sorted variant to MSL for
   throughput on dense problems.
5. **2D / 3D**: extend kernels and FFT calls.
6. **Type 2, then Type 3**: type 2 mostly mirrors type 1; type 3 needs
   the rescaling/correction structure plus careful FFT-shape choices.
7. **Hardening**: error paths (Metal command buffer status), pipeline
   state caching across plans, profiling with Instruments / Metal
   Frame Capture.
8. **Document fp64 status**: ship fp32-only on Metal initially; route
   fp64 to CPU finufft with a clear runtime warning.

## Risk register

- **MPSGraph FFT size restrictions** — verify before committing to it.
- **fp64 performance** — non-negotiable cliff; communicate it clearly.
- **Type-3 sizing** — same 2/3/5-smooth assumptions as elsewhere; the
  Metal FFT must accept them or we coarsen.
- **CI cost / availability** — macOS-arm64 runners are a recurring
  cost item; budget for it or run nightly.
- **Atomic-float corner cases** — Metal atomic_float exists but verify
  ordering semantics match what the CUDA spreader assumes (relaxed vs
  acq-rel).
- **Library distribution** — shipping `.metallib` blobs in wheels/conda
  packages requires per-arch builds and signing; non-trivial.

## Recommended first step

Spike a single-source proof-of-concept: 1D type-1 fp32, MPSGraph FFT,
naive MSL atomic-add spreader, host code in metal-cpp, CMake-gated
build. Goal is end-to-end correctness against CPU finufft on one test
case in <500 lines, not performance. Once that runs, the rest is
incremental.
