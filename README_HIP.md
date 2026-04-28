# AMD/HIP backend (`libhipfinufft`)

This document describes the AMD/HIP backend of FINUFFT, built in parallel
to the existing CUDA backend (`libcufinufft`). The two are independent --
both can be enabled in the same build, and they install side-by-side.

For build instructions see [`docs/install_hip.rst`](docs/install_hip.rst).

## TL;DR

```bash
cmake -S . -B build -DFINUFFT_USE_HIP=ON -DFINUFFT_USE_CPU=OFF
cmake --build build -j
```

## Repository layout

The HIP backend mirrors the CUDA layout one-for-one:

```
finufft/
├── src/
│   ├── cuda/                  # CUDA backend (untouched)
│   └── hip/                   # AMD backend (parallel tree)
├── include/
│   ├── cufinufft.h            # CUDA public C header
│   ├── cufinufft_opts.h
│   ├── cufinufft/             # CUDA internal headers
│   ├── hipfinufft.h           # AMD public C header
│   ├── hipfinufft_opts.h
│   └── hipfinufft/            # AMD internal headers
├── test/
│   ├── cuda/                  # CUDA tests + ctest entries
│   └── hip/                   # AMD tests + ctest entries
├── examples/
│   ├── cuda/
│   └── hip/
├── perftest/
│   ├── cuda/
│   └── hip/
├── cmake/
│   ├── cuda_setup.cmake       # finds CUDAToolkit, sm autodetect
│   └── hip_setup.cmake        # finds hip/hipfft/rocthrust, gfx autodetect
└── tools/
    └── hipfinufft/
        └── cuda_to_hip.py     # deterministic CUDA→HIP translator
```

## How the port is maintained

The AMD tree is generated from the CUDA tree by a deterministic translator
script: `tools/hipfinufft/cuda_to_hip.py`. The script does
hipify-style token substitution plus a small number of structural rewrites
(strip NVIDIA-only shared-memory bank hints, replace `__CUDA_ARCH__` guards,
rename file basenames). It is **idempotent**: running it twice produces the
same output as running it once.

Workflow when upstream CUDA changes:

```bash
# 1. Regenerate the HIP tree
python3 tools/hipfinufft/cuda_to_hip.py

# 2. Audit any residual CUDA-flavored tokens (should be empty)
python3 tools/hipfinufft/cuda_to_hip.py --report

# 3. In CI, gate on:
python3 tools/hipfinufft/cuda_to_hip.py --check
```

The four `CMakeLists.txt` files in the HIP tree are **hand-written**
(translator skips them by name), because they use different CMake target
properties (`HIP_ARCHITECTURES` instead of `CUDA_ARCHITECTURES`,
`hip::host`/`hip::hipfft`/`roc::rocthrust` instead of `CUDA::cudart`/etc).

The small number of "this is structural, not a token" decisions are all
documented as named regex `Rule`s in the translator. Find them by searching
for `COMPLEX_RULES`.

## Library coexistence

`libcufinufft` and `libhipfinufft` expose **different** C symbols
(`cufinufft_*` vs `hipfinufft_*`) and **different** error codes
(`FINUFFT_ERR_CUDA_FAILURE = 15` vs `FINUFFT_ERR_HIP_FAILURE = 27`). They
are siblings, not replacements. An application that wants to dispatch to
whichever GPU is present at runtime can link both.

## What's *not* ported

- **Python bindings** (`python/cufinufft/`)
- **MATLAB bindings** (`matlab/cufinufft.{cu,mw}`)
- **Fortran tests / examples** (no GPU Fortran path exists upstream anyway)
- **Julia bindings** (live in a separate repo; would need their own port)

These are out of scope for the initial port. Each has its own codegen flow
and would need a focused effort.

## Status

This is a **mechanical first pass**. Everything compiles structurally
(modulo the standard porting risks called out below), and numerical
results should match the CUDA backend within tolerance. **None of this
has been hardware-validated by me.** I don't have ROCm in the build
environment where the port was authored.

Known caveats live in [`docs/install_hip.rst`](docs/install_hip.rst#known-limitations);
the high-impact ones are:

1. **Performance defaults are NVIDIA-tuned.** Bin sizes and `gpu_np`
   were chosen for 32-thread warps. AMD CDNA wavefronts are 64-wide;
   tuning for these is follow-up work.
2. **`__ldg` and friends fall back to plain loads on AMD.** Correctness
   is preserved; perf opportunity left on the table.
3. **NVIDIA-only shared-memory bank-width hint** (`hipFuncSetSharedMemConfig`)
   is stripped on HIP since AMD has no equivalent.

## Validating on real hardware

Once you have a ROCm box available, the canonical smoke test is:

```bash
cmake -S . -B build -DFINUFFT_USE_HIP=ON -DFINUFFT_USE_CPU=OFF \
    -DFINUFFT_BUILD_TESTS=ON -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected to run: ~80 ctest entries covering 1D/2D/3D × type-1/2/3 ×
single/double × gpu_method ∈ {auto, GM, SM, OD, block} × upsamp ∈
{2.0, 1.25, default}.
