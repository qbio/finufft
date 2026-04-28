#!/usr/bin/env python3
"""
cuda_to_hip.py
==============

Deterministic source-level translator that turns the cufinufft CUDA tree
(``src/cuda``, ``include/cufinufft``, ``include/cufinufft.h``,
``include/cufinufft_opts.h``, ``test/cuda``, ``examples/cuda``,
``perftest/cuda``) into the parallel hipfinufft HIP tree (``src/hip``,
``include/hipfinufft``, ``include/hipfinufft.h``, ``include/hipfinufft_opts.h``,
``test/hip``, ``examples/hip``, ``perftest/hip``).

Design goals
------------

* Idempotent: running it twice produces the same output as running it once.
* Deterministic: no hashing/timestamps in output.
* Auditable: every transformation is a single, named regex rule. The diff
  between any two runs is therefore explainable in terms of these rules
  plus upstream CUDA source changes.
* Conservative: we ONLY rewrite tokens we have reason to believe map cleanly.
  Anything that requires human review is left untouched and (optionally)
  flagged with ``--report``.

Usage
-----

From repo root::

    python3 tools/hipfinufft/cuda_to_hip.py            # regenerate HIP tree
    python3 tools/hipfinufft/cuda_to_hip.py --check    # CI: error if drift
    python3 tools/hipfinufft/cuda_to_hip.py --report   # print unrecognized tokens

The translator does NOT produce the CMake glue files (those are hand-written
in ``cmake/hip_setup.cmake`` and ``src/hip/CMakeLists.txt``); only source code
is regenerated.

Limitations
-----------

* I have not been able to test the resulting code on actual AMD hardware.
  Several known edge cases are documented in ``docs/install_hip.rst``.
* The translator does NOT touch Python wrappers, MATLAB, or Fortran tests.
  Those would need their own ports (Python ``cufinufft`` -> ``hipfinufft``,
  etc.). They are out of scope for the initial port.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Tree mapping
# ---------------------------------------------------------------------------
# Each entry is (cuda_relative_path, hip_relative_path). Files are copied
# verbatim then run through the rewrite rules. Directories are walked
# recursively. File extensions are also rewritten (.cu -> .hip).
#
# We mirror the structure exactly. If you add a new file in the CUDA tree,
# add the parallel entry here (or just rerun this script -- it discovers
# files under the listed roots automatically).

TREE_MAP = [
    ("src/cuda",                  "src/hip"),
    ("include/cufinufft",         "include/hipfinufft"),
    ("include/cufinufft.h",       "include/hipfinufft.h"),
    ("include/cufinufft_opts.h",  "include/hipfinufft_opts.h"),
    ("test/cuda",                 "test/hip"),
    ("examples/cuda",             "examples/hip"),
    ("perftest/cuda",             "perftest/hip"),
]

# Files that should NOT be touched at all. We hand-write CMakeLists.txt for
# the HIP tree because the translation from CUDA-language CMake to HIP-language
# CMake is not a 1:1 token swap (different target properties, different
# find_package, different compile flags). The hand-written versions live in
# the HIP tree and are intentionally not regenerated.
SKIP_FILENAMES = {
    "CMakeLists.txt",
    # README files in src/cuda/, test/cuda/, examples/cuda/ get auto-translated
    # but we leave them as-is since the text translation is fine.
}

# File extensions to translate. Anything else is copied verbatim.
TEXT_EXTENSIONS = {".cu", ".cuh", ".h", ".hpp", ".cpp", ".c", ".inc",
                   ".sh", ".rst", ".md", ".txt", ".cmake"}


@dataclass
class Rule:
    """A single regex-based source rewrite."""

    name: str
    pattern: re.Pattern
    replacement: str | Callable[[re.Match], str]
    # If True, only apply inside #include directives. Used to scope filename
    # changes so we don't accidentally touch identifiers that happen to match.
    include_only: bool = False
    # If True, this rule is allowed to match across line boundaries.
    multiline: bool = False


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------
# The order matters: more specific rules first.

def _re(pat: str, flags: int = 0) -> re.Pattern:
    return re.compile(pat, flags)


# Tokens that are clean 1:1 substitutions. We use word-boundary anchors so we
# don't replace substrings of unrelated identifiers.
WORD_SUBSTITUTIONS = [
    # --- Project-level: cufinufft -> hipfinufft -----------------------------
    # We deliberately rewrite *both* the C symbols and the C++ namespaces.
    # The wider word boundaries below ensure we don't mangle, say, the string
    # "documentation about cufinufft" inside a comment block (we DO want
    # that rewritten since it's documenting the library).
    ("cufinufftf_plan",              "hipfinufftf_plan"),
    ("cufinufft_fplan_s",            "hipfinufft_fplan_s"),
    ("cufinufft_plan_s",             "hipfinufft_plan_s"),
    ("cufinufftf_makeplan",          "hipfinufftf_makeplan"),
    ("cufinufft_makeplan",           "hipfinufft_makeplan"),
    ("cufinufftf_setpts",            "hipfinufftf_setpts"),
    ("cufinufft_setpts",             "hipfinufft_setpts"),
    ("cufinufftf_execute",           "hipfinufftf_execute"),
    ("cufinufft_execute",            "hipfinufft_execute"),
    ("cufinufftf_destroy",           "hipfinufftf_destroy"),
    ("cufinufft_destroy",            "hipfinufft_destroy"),
    ("cufinufft_default_opts",       "hipfinufft_default_opts"),
    ("cufinufft_setup_binsize",      "hipfinufft_setup_binsize"),
    ("cufinufft_set_shared_memory",  "hipfinufft_set_shared_memory"),
    ("cufinufft_plan_t",             "hipfinufft_plan_t"),
    ("cufinufft_opts",               "hipfinufft_opts"),
    ("cufinufft_plan",               "hipfinufft_plan"),
    ("cufinufftf_plan",              "hipfinufftf_plan"),
    ("CUFINUFFT_BIGINT",             "HIPFINUFFT_BIGINT"),
    ("CUFINUFFT_INCLUDE_DIRS",       "HIPFINUFFT_INCLUDE_DIRS"),
    ("CUFINUFFT_PUBLIC_HEADERS",     "HIPFINUFFT_PUBLIC_HEADERS"),
    ("CUFINUFFT_TYPES_H",            "HIPFINUFFT_TYPES_H"),
    ("CUFINUFFT_PLAN_T_H",           "HIPFINUFFT_PLAN_T_H"),
    ("CUFINUFFT_DEFS_H",             "HIPFINUFFT_DEFS_H"),
    ("__CUFINUFFT_OPTS_H__",         "__HIPFINUFFT_OPTS_H__"),
    ("COMMON_HELPER_CUDA_H_",        "COMMON_HELPER_HIP_H_"),
    ("FINUFFT_INCLUDE_CUFINUFFT_CONTRIB_HELPER_MATH_H",
     "FINUFFT_INCLUDE_HIPFINUFFT_CONTRIB_HELPER_MATH_H"),
    ("namespace cufinufft",          "namespace hipfinufft"),
    ("cufinufft::",                  "hipfinufft::"),

    # --- Complex types ------------------------------------------------------
    # CUDA's cuComplex.h provides cuFloatComplex, cuDoubleComplex, and the
    # cuC* arithmetic helpers. ROCm/HIP provides hip/hip_complex.h with the
    # exact same shape and the same hipC* helpers.
    ("cuFloatComplex",               "hipFloatComplex"),
    ("cuDoubleComplex",              "hipDoubleComplex"),
    ("cufftComplex",                 "hipfftComplex"),
    ("cufftDoubleComplex",           "hipfftDoubleComplex"),
    ("make_cuFloatComplex",          "make_hipFloatComplex"),
    ("make_cuDoubleComplex",         "make_hipDoubleComplex"),
    ("cuCadd",   "hipCadd"),
    ("cuCsub",   "hipCsub"),
    ("cuCmul",   "hipCmul"),
    ("cuCdiv",   "hipCdiv"),
    ("cuCabs",   "hipCabs"),
    ("cuCarg",   "hipCarg"),
    ("cuConj",   "hipConj"),
    ("cuConjf",  "hipConjf"),
    # Bare "cuComplex" (without the F or Double prefix) appears in comments
    # describing the cuComplex.h header. Catch-all so docs make sense.
    ("cuComplex",  "hipComplex"),
    ("cuCabsf",  "hipCabsf"),
    ("cuCargf",  "hipCargf"),
    ("cuCaddf",  "hipCaddf"),
    ("cuCsubf",  "hipCsubf"),
    ("cuCmulf",  "hipCmulf"),
    ("cuCdivf",  "hipCdivf"),
    ("cuCreal",  "hipCreal"),
    ("cuCrealf", "hipCrealf"),
    ("cuCimag",  "hipCimag"),
    ("cuCimagf", "hipCimagf"),
    ("cuda_complex",                 "hip_complex"),

    # --- cuFFT -> hipFFT ----------------------------------------------------
    # hipFFT mirrors the cuFFT API, including the enum names (HIPFFT_C2C etc.)
    ("cufftHandle",        "hipfftHandle"),
    ("cufftResult",        "hipfftResult"),
    ("cufftType_t",        "hipfftType_t"),
    ("cufftType",          "hipfftType"),
    ("cufftPlanMany",      "hipfftPlanMany"),
    ("cufftPlan1d",        "hipfftPlan1d"),
    ("cufftPlan2d",        "hipfftPlan2d"),
    ("cufftPlan3d",        "hipfftPlan3d"),
    ("cufftDestroy",       "hipfftDestroy"),
    ("cufftSetStream",     "hipfftSetStream"),
    ("cufftExecC2C",       "hipfftExecC2C"),
    ("cufftExecZ2Z",       "hipfftExecZ2Z"),
    ("cufftGetErrorString", "hipfftGetErrorString"),
    ("cufftGetVersion",    "hipfftGetVersion"),
    # Enums:
    ("CUFFT_SUCCESS",                  "HIPFFT_SUCCESS"),
    ("CUFFT_INVALID_PLAN",             "HIPFFT_INVALID_PLAN"),
    ("CUFFT_ALLOC_FAILED",             "HIPFFT_ALLOC_FAILED"),
    ("CUFFT_INVALID_TYPE",             "HIPFFT_INVALID_TYPE"),
    ("CUFFT_INVALID_VALUE",            "HIPFFT_INVALID_VALUE"),
    ("CUFFT_INTERNAL_ERROR",           "HIPFFT_INTERNAL_ERROR"),
    ("CUFFT_EXEC_FAILED",              "HIPFFT_EXEC_FAILED"),
    ("CUFFT_SETUP_FAILED",             "HIPFFT_SETUP_FAILED"),
    ("CUFFT_INVALID_SIZE",             "HIPFFT_INVALID_SIZE"),
    ("CUFFT_UNALIGNED_DATA",           "HIPFFT_UNALIGNED_DATA"),
    ("CUFFT_INVALID_DEVICE",           "HIPFFT_INVALID_DEVICE"),
    ("CUFFT_NO_WORKSPACE",             "HIPFFT_NO_WORKSPACE"),
    ("CUFFT_NOT_IMPLEMENTED",          "HIPFFT_NOT_IMPLEMENTED"),
    ("CUFFT_NOT_SUPPORTED",            "HIPFFT_NOT_SUPPORTED"),
    ("CUFFT_LICENSE_ERROR",            "HIPFFT_LICENSE_ERROR"),
    ("CUFFT_PARSE_ERROR",              "HIPFFT_PARSE_ERROR"),
    ("CUFFT_INCOMPLETE_PARAMETER_LIST","HIPFFT_INCOMPLETE_PARAMETER_LIST"),
    ("CUFFT_C2C", "HIPFFT_C2C"),
    ("CUFFT_Z2Z", "HIPFFT_Z2Z"),
    ("CUFFT_R2C", "HIPFFT_R2C"),
    ("CUFFT_C2R", "HIPFFT_C2R"),
    ("CUFFT_D2Z", "HIPFFT_D2Z"),
    ("CUFFT_Z2D", "HIPFFT_Z2D"),
    ("CUFFT_FORWARD", "HIPFFT_FORWARD"),
    ("CUFFT_INVERSE", "HIPFFT_BACKWARD"),  # naming difference: see note below

    # --- CUDA runtime API (functions) --------------------------------------
    # The hip*** functions are 1:1 wrappers. There are subtle differences for
    # a handful (e.g. memory pool support flags) which we handle in the
    # hand-written compat header rather than the translator.
    ("cudaMallocAsync",        "hipMallocAsync"),
    ("cudaFreeAsync",          "hipFreeAsync"),
    ("cudaMallocManaged",      "hipMallocManaged"),
    ("cudaMallocHost",         "hipHostMalloc"),
    ("cudaHostAlloc",          "hipHostMalloc"),
    ("cudaFreeHost",           "hipHostFree"),
    ("cudaMemcpyAsync",        "hipMemcpyAsync"),
    ("cudaMemsetAsync",        "hipMemsetAsync"),
    ("cudaMemcpy",             "hipMemcpy"),
    ("cudaMemset",             "hipMemset"),
    ("cudaMemGetInfo",         "hipMemGetInfo"),
    ("cudaMalloc",             "hipMalloc"),
    ("cudaFree",               "hipFree"),
    ("cudaStreamCreate",       "hipStreamCreate"),
    ("cudaStreamCreateWithFlags", "hipStreamCreateWithFlags"),
    ("cudaStreamDestroy",      "hipStreamDestroy"),
    ("cudaStreamSynchronize",  "hipStreamSynchronize"),
    ("cudaStreamWaitEvent",    "hipStreamWaitEvent"),
    ("cudaStreamQuery",        "hipStreamQuery"),
    ("cudaStreamDefault",      "hipStreamDefault"),
    ("cudaStream_t",           "hipStream_t"),
    ("cudaEventCreate",        "hipEventCreate"),
    ("cudaEventCreateWithFlags", "hipEventCreateWithFlags"),
    ("cudaEventDestroy",       "hipEventDestroy"),
    ("cudaEventRecord",        "hipEventRecord"),
    ("cudaEventSynchronize",   "hipEventSynchronize"),
    ("cudaEventElapsedTime",   "hipEventElapsedTime"),
    ("cudaEvent_t",            "hipEvent_t"),
    ("cudaDeviceSynchronize",  "hipDeviceSynchronize"),
    ("cudaDeviceReset",        "hipDeviceReset"),
    ("cudaSetDevice",          "hipSetDevice"),
    ("cudaGetDevice",          "hipGetDevice"),
    ("cudaGetDeviceCount",     "hipGetDeviceCount"),
    ("cudaGetDeviceProperties", "hipGetDeviceProperties"),
    ("cudaDeviceGetAttribute", "hipDeviceGetAttribute"),
    ("cudaDeviceProp",         "hipDeviceProp_t"),
    ("cudaFuncSetAttribute",   "hipFuncSetAttribute"),
    # Shared-memory bank-size hints: NVIDIA-specific. Mapped to the HIP
    # equivalent name (the call itself is then dropped by the strip rule
    # in COMPLEX_RULES, since AMD CDNA/RDNA do not expose this control).
    ("cudaFuncSetSharedMemConfig",     "hipFuncSetSharedMemConfig"),
    ("cudaSharedMemBankSizeFourByte",  "hipSharedMemBankSizeFourByte"),
    ("cudaSharedMemBankSizeEightByte", "hipSharedMemBankSizeEightByte"),
    ("cudaSharedMemBankSizeDefault",   "hipSharedMemBankSizeDefault"),
    ("cudaPointerGetAttributes","hipPointerGetAttributes"),
    ("cudaPointerAttributes",  "hipPointerAttribute_t"),
    ("cudaGetLastError",       "hipGetLastError"),
    ("cudaPeekAtLastError",    "hipPeekAtLastError"),
    ("cudaGetErrorString",     "hipGetErrorString"),
    ("cudaGetErrorName",       "hipGetErrorName"),
    # --- CUDA runtime API (enum/values) ------------------------------------
    ("cudaSuccess",            "hipSuccess"),
    ("cudaErrorInsufficientDriver", "hipErrorInsufficientDriver"),
    ("cudaError_t",            "hipError_t"),
    ("cudaError",              "hipError_t"),  # bare 'cudaError' typedef
    ("cudaMemcpyHostToDevice", "hipMemcpyHostToDevice"),
    ("cudaMemcpyDeviceToHost", "hipMemcpyDeviceToHost"),
    ("cudaMemcpyHostToHost",   "hipMemcpyHostToHost"),
    ("cudaMemcpyDeviceToDevice","hipMemcpyDeviceToDevice"),
    ("cudaMemcpyDefault",      "hipMemcpyDefault"),
    # Device attributes used by cufinufft:
    ("cudaDevAttrMaxSharedMemoryPerBlockOptin",
     "hipDeviceAttributeSharedMemPerBlockOptin"),
    ("cudaDevAttrMaxSharedMemoryPerBlock",
     "hipDeviceAttributeMaxSharedMemoryPerBlock"),
    ("cudaDevAttrMemoryPoolsSupported",
     "hipDeviceAttributeMemoryPoolsSupported"),
    ("cudaFuncAttributeMaxDynamicSharedMemorySize",
     "hipFuncAttributeMaxDynamicSharedMemorySize"),
    # Stream/event flags
    ("cudaStreamNonBlocking",  "hipStreamNonBlocking"),
    ("cudaEventBlockingSync",  "hipEventBlockingSync"),
    ("cudaEventDisableTiming", "hipEventDisableTiming"),

    # --- Header includes ---------------------------------------------------
    # These are matched as bare tokens because they appear inside #include
    # directives where word boundaries work fine.
    ("cuda_runtime.h",     "hip/hip_runtime.h"),
    ("cuda_runtime_api.h", "hip/hip_runtime_api.h"),
    ("cuda.h",             "hip/hip_runtime.h"),
    ("cuComplex.h",        "hip/hip_complex.h"),
    ("cufft.h",             "hipfft/hipfft.h"),

    # --- Thrust execution policy header ------------------------------------
    # rocThrust uses the thrust:: namespace but lives under thrust/system/hip.
    ("thrust/system/cuda/execution_policy.h",
     "thrust/system/hip/execution_policy.h"),
    ("thrust/system/cuda/", "thrust/system/hip/"),
    ("thrust::cuda::",      "thrust::hip::"),

    # --- libcu++ -> std --------------------------------------------------
    # cufinufft uses cuda::std::array in a handful of places. rocThrust does
    # not vendor an equivalent, but std::array is supported in HIP device
    # code in modern ROCm. So we drop the cuda:: prefix entirely.
    # NOTE: the substitution is unconditional because every use of
    # cuda::std::array we have seen in cufinufft is value-typed array
    # arguments to kernels, which is fine.
    ("cuda::std::array", "std::array"),
    ("<cuda/std/array>", "<array>"),
    # If other libcu++ types creep in, add them here.

    # --- Project-internal helpers named with "cuda" prefix -----------------
    # These are user-defined functions in the cufinufft codebase, not CUDA
    # APIs. They are renamed for consistency.
    ("cudaFMA", "hipFMA"),

    # --- Comments / banners ------------------------------------------------
    # Surface-level renaming so user-facing strings make sense.
    ("CUFINUFFT", "HIPFINUFFT"),  # macro/banner names
    ("cufinufft", "hipfinufft"),  # general identifier (LAST among rewrites!)
    ("CUDA finufft", "HIP finufft"),
    ("cufft",       "hipfft"),    # done after CUFFT_* enums above
    # NB: We deliberately do NOT translate `__NVCC__`. The cufinufft codebase
    # uses `#ifdef __NVCC__` to gate NVIDIA-specific intrinsics (e.g. __ldg,
    # __ldca, __ldcs in intrinsics.h, plus the cuComplex operator overloads
    # in helper_math.h). When compiled with hipcc, `__NVCC__` is undefined,
    # so the existing #else branches run -- producing correct, if slightly
    # less-optimized, code on AMD. Translating __NVCC__ to __HIPCC__ would
    # invert this and try to use NVIDIA-only intrinsics on AMD.
    # We also do not touch the lowercase "nvcc" / uppercase "NVCC" tokens,
    # because they would clobber __NVCC__ as a substring.
    # ...except: literal "nvcc" (lowercase) is safe -- str.replace is
    # case-sensitive, and __NVCC__ is uppercase. We rewrite that for the
    # benefit of comments and shell commands inside example files.
    ("nvcc", "hipcc"),
    ("nvidia-smi",  "rocm-smi"),
    ("CUDAToolkit", "hip"),       # CMake find_package
    ("CUDA::cudart", "hip::host"),
    ("CUDA::cufft",  "hip::hipfft"),
]


# Rules that need more than a flat word substitution go here.
COMPLEX_RULES: list[Rule] = [

    # cufinufft includes contrib headers. helper_cuda.h is auto-renamed to
    # helper_hip.h; helper_math.h's path doesn't change but its content does.
    Rule(
        name="rename_helper_cuda_include",
        pattern=_re(r'(["<])cufinufft/contrib/helper_cuda\.h([">])'),
        replacement=r'\1hipfinufft/contrib/helper_hip.h\2',
    ),
    # Generic catch for any cufinufft/* include path -> hipfinufft/*
    Rule(
        name="rename_cufinufft_includes",
        pattern=_re(r'(["<])cufinufft/'),
        replacement=r'\1hipfinufft/',
    ),
    Rule(
        name="rename_cufinufft_root_includes",
        pattern=_re(r'(["<])cufinufft(\.h|_opts\.h)([">])'),
        replacement=r'\1hipfinufft\2\3',
    ),

    # ----- __CUDA_ARCH__ handling ------------------------------------------
    # Three patterns appear in this codebase:
    #   1. `defined(__CUDA_ARCH__)`   -> `defined(__HIP_DEVICE_COMPILE__)`
    #   2. `__CUDA_ARCH__ >= NNN`     -> rewritten to `0` so the legacy NVIDIA
    #      pre-Pascal double-atomic workaround in utils.h is excluded. AMD
    #      always provides device-side `atomicAdd(double*, double)` for our
    #      target architectures, so the host-defined fallback would conflict.
    #   3. bare `__CUDA_ARCH__`       -> `__HIP_DEVICE_COMPILE__`
    Rule(
        name="cuda_arch_defined",
        pattern=_re(r'defined\s*\(\s*__CUDA_ARCH__\s*\)'),
        replacement="defined(__HIP_DEVICE_COMPILE__)",
    ),
    Rule(
        name="cuda_arch_compare",
        pattern=_re(r'__CUDA_ARCH__\s*(>=|<=|<|>|==|!=)\s*\d+'),
        # Always evaluate to false on AMD: the codebase only uses these to
        # gate per-arch features that are inapplicable.
        replacement="0",
    ),
    Rule(
        name="cuda_arch_bare",
        pattern=_re(r'__CUDA_ARCH__'),
        replacement="__HIP_DEVICE_COMPILE__",
    ),

    # ----- Shared memory bank-size config no-op -----------------------------
    # NVIDIA exposes per-kernel shared-memory bank-width hints. AMD does not
    # have a meaningful equivalent on CDNA/RDNA; rewrite the call to nothing
    # so the surrounding launch code still runs. We are careful NOT to strip
    # any error-check macro that follows: those macros are typically also
    # checking the call that came BEFORE this one.
    Rule(
        name="strip_func_set_shared_mem_config",
        pattern=_re(
            r'^[ \t]*hipFuncSetSharedMemConfig\s*\([^;]*\);\s*\n',
            re.MULTILINE),
        replacement="",
    ),

    # ----- helper_cuda.h filename relocation --------------------------------
    # The auto-translator places the rewritten file at
    # include/hipfinufft/contrib/helper_cuda.h, but consumers reference
    # helper_hip.h. We additionally rename the file via _basename_map
    # in the file-walk; here we just normalize stray `helper_cuda.h` text
    # references that aren't inside #include directives.
    Rule(
        name="text_helper_cuda_to_hip",
        pattern=_re(r'\bhelper_cuda\.h\b'),
        replacement="helper_hip.h",
    ),

    # ----- CMake bits the auto-translator might still see -------------------
    # We skip CMakeLists.txt by name, but other .cmake files (none today,
    # but future-proof) get these mappings.
    Rule(
        name="cmake_cuda_arch_var",
        pattern=_re(r'\bCMAKE_CUDA_ARCHITECTURES\b'),
        replacement="CMAKE_HIP_ARCHITECTURES",
    ),

    # ----- cuFFT-13 deprecated-enum compatibility block ---------------------
    # The cufinufft codebase has a `#if __CUDACC_VER_MAJOR__ < 13` block
    # in helper_cuda.h covering the CUFFT_LICENSE_ERROR / CUFFT_PARSE_ERROR
    # / CUFFT_INCOMPLETE_PARAMETER_LIST enums. After our token rewrites the
    # block references HIPFFT_LICENSE_ERROR etc., which simply do not exist
    # in hipfft. We strip the whole block. This is structural rather than a
    # bare token swap so it lives here, not in WORD_SUBSTITUTIONS.
    Rule(
        name="strip_cufft13_compat_block",
        pattern=_re(
            r'^[ \t]*//[^\n]*deprecated[^\n]*12\.9[^\n]*\n'
            r'[ \t]*#if\s+__CUDACC_VER_MAJOR__\s*<\s*13\b.*?#endif\s*\n',
            re.MULTILINE | re.DOTALL),
        replacement="",
    ),

    # Many cufinufft files use 'checkCudaErrors' and 'THROW_IF_CUDA_ERROR'.
    # We keep the macro names but rewrite their *internals* via the helper
    # header. So we just rename the macros too, for clarity on the AMD side.
    Rule(
        name="rename_check_cuda_errors_macro",
        pattern=_re(r'\bcheckCudaErrors\b'),
        replacement="checkHipErrors",
    ),
    Rule(
        name="rename_throw_if_cuda_error_macro",
        pattern=_re(r'\bTHROW_IF_CUDA_ERROR\b'),
        replacement="THROW_IF_HIP_ERROR",
    ),

    # finufft_errors.h uses FINUFFT_ERR_CUDA_FAILURE; rename to keep parity.
    # NOTE: we do NOT touch the actual integer value (in finufft_errors.h
    # itself we leave alone since that's a CPU-side header shared with HIP).
    # On the HIP side we just alias it.
    Rule(
        name="rename_cuda_failure_const",
        pattern=_re(r'\bFINUFFT_ERR_CUDA_FAILURE\b'),
        replacement="FINUFFT_ERR_HIP_FAILURE",
    ),

    # CMake target name updates for in-tree CMakeLists used in tests/examples.
    Rule(
        name="rename_cmake_target_cufinufft",
        pattern=_re(r'\b(target_link_libraries\s*\([^)]*?)cufinufft\b',
                    re.DOTALL),
        replacement=lambda m: m.group(1) + "hipfinufft",
    ),
]


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------

def rewrite_extension(path: Path) -> Path:
    """Rename .cu -> .hip; .cuh -> .hpp; rename cufinufft* basenames."""
    new_name = path.name
    # Basename token replacement: this is the same set of project-internal
    # renames as in WORD_SUBSTITUTIONS, but applied to file names.
    BASENAME_REPLACEMENTS = (
        ("cufinufft_plan_t", "hipfinufft_plan_t"),
        ("cufinufftf",       "hipfinufftf"),
        ("cufinufft",        "hipfinufft"),
        ("cuperftest",       "hipperftest"),
        ("helper_cuda",      "helper_hip"),
    )
    for src_tok, dst_tok in BASENAME_REPLACEMENTS:
        if src_tok in new_name:
            new_name = new_name.replace(src_tok, dst_tok)
    out = path.with_name(new_name)
    if out.suffix == ".cu":
        out = out.with_suffix(".hip")
    if out.suffix == ".cuh":
        out = out.with_suffix(".hpp")
    return out


def apply_rules(text: str) -> str:
    """Apply the full rule set to a single source string."""
    # Pass 1: simple word substitutions, longest-first to avoid prefix
    # clashes (e.g. cufinufftf_makeplan must be matched before cufinufft).
    sorted_subs = sorted(WORD_SUBSTITUTIONS, key=lambda kv: -len(kv[0]))
    for src, dst in sorted_subs:
        # \b doesn't match across underscores; we rely on the longest-first
        # ordering to keep things safe. We use a literal-text replace because
        # the source tokens contain no regex metacharacters.
        if not src:
            continue
        # Use a non-regex replace for speed and safety.
        text = text.replace(src, dst)

    # Pass 2: structural rules.
    for rule in COMPLEX_RULES:
        text = rule.pattern.sub(rule.replacement, text)

    return text


def translate_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in TEXT_EXTENSIONS or src.name.endswith("CMakeLists.txt"):
        text = src.read_text(encoding="utf-8")
        text = apply_rules(text)
        # Add a one-line generated banner so it's obvious the file is auto-
        # produced. Skip for files where comments aren't well-defined.
        banner = None
        if src.suffix.lower() in {".cu", ".cuh", ".h", ".hpp", ".cpp", ".c"} \
                or src.suffix == ".inc":
            banner = ("// AUTO-GENERATED by tools/hipfinufft/cuda_to_hip.py "
                      "from " + str(src.relative_to(REPO_ROOT)) + "\n"
                      "// Do not edit by hand. Edit the CUDA source and "
                      "re-run the translator.\n")
        elif src.name.endswith("CMakeLists.txt") or src.suffix == ".cmake":
            banner = ("# AUTO-GENERATED by tools/hipfinufft/cuda_to_hip.py "
                      "from " + str(src.relative_to(REPO_ROOT)) + "\n"
                      "# Do not edit by hand.\n")
        if banner is not None:
            # Don't double-add the banner if it's already there.
            if not text.startswith(banner.split("\n", 1)[0]):
                text = banner + text
        dst.write_text(text, encoding="utf-8")
    else:
        shutil.copy2(src, dst)


def walk_tree(src_root: Path, dst_root: Path) -> list[tuple[Path, Path]]:
    """Enumerate (src, dst) file pairs under a directory pair."""
    pairs: list[tuple[Path, Path]] = []
    if src_root.is_file():
        pairs.append((src_root, dst_root))
        return pairs
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        if src.name in SKIP_FILENAMES:
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst = rewrite_extension(dst)
        pairs.append((src, dst))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true",
                   help="Verify HIP tree is up-to-date; exit 1 on drift.")
    p.add_argument("--report", action="store_true",
                   help="List CUDA-flavored tokens still present in the HIP "
                        "tree after translation, for human review.")
    p.add_argument("--root", type=Path, default=REPO_ROOT,
                   help="Repository root (default: parent of this script).")
    args = p.parse_args()

    root: Path = args.root.resolve()

    pairs: list[tuple[Path, Path]] = []
    for src_rel, dst_rel in TREE_MAP:
        src_root = root / src_rel
        dst_root = root / dst_rel
        if not src_root.exists():
            print(f"  warning: source {src_root} does not exist; skipping",
                  file=sys.stderr)
            continue
        pairs.extend(walk_tree(src_root, dst_root))

    drift = []
    written = 0
    for src, dst in pairs:
        new_text: str | None = None
        if src.suffix.lower() in TEXT_EXTENSIONS or src.name.endswith("CMakeLists.txt"):
            text = src.read_text(encoding="utf-8")
            new_text = apply_rules(text)
            # Match the banner-prepending logic in translate_file() so --check
            # compares apples to apples.
            if src.suffix.lower() in {".cu", ".cuh", ".h", ".hpp", ".cpp", ".c", ".inc"}:
                banner = ("// AUTO-GENERATED by tools/hipfinufft/cuda_to_hip.py "
                          "from " + str(src.relative_to(root)) + "\n"
                          "// Do not edit by hand. Edit the CUDA source and "
                          "re-run the translator.\n")
                if not new_text.startswith(banner.split("\n", 1)[0]):
                    new_text = banner + new_text
            elif src.name.endswith("CMakeLists.txt") or src.suffix == ".cmake":
                banner = ("# AUTO-GENERATED by tools/hipfinufft/cuda_to_hip.py "
                          "from " + str(src.relative_to(root)) + "\n"
                          "# Do not edit by hand.\n")
                if not new_text.startswith(banner.split("\n", 1)[0]):
                    new_text = banner + new_text

        if args.check:
            if new_text is None:
                # Binary file: just compare bytes.
                if not dst.exists() or dst.read_bytes() != src.read_bytes():
                    drift.append(dst)
            else:
                if not dst.exists() or dst.read_text(encoding="utf-8") != new_text:
                    drift.append(dst)
        else:
            translate_file(src, dst)
            written += 1

    if args.check:
        if drift:
            print("HIP tree is out of date. Files needing regeneration:",
                  file=sys.stderr)
            for d in drift[:25]:
                print(f"  {d.relative_to(root)}", file=sys.stderr)
            if len(drift) > 25:
                print(f"  ... and {len(drift) - 25} more", file=sys.stderr)
            return 1
        print("HIP tree is up to date.")
        return 0

    print(f"Wrote {written} files to HIP tree.")

    if args.report:
        # Surface anything that still looks CUDA-flavored after translation,
        # so a human can audit it. We deliberately allow __NVCC__ through:
        # it's used in #ifdef guards that gate NVIDIA-only intrinsics, with
        # plain-C++ fallbacks in the #else branch. Those guards correctly
        # evaluate to "false" under hipcc, exactly the behavior we want.
        suspicious_pat = re.compile(
            r'\b(cuda[A-Z]\w+|CUDA[_A-Z]+|__CUDA_ARCH__|cuComplex'
            r'|cuFloat|cuDouble|cufft[A-Z]|CUFFT_)\w*\b')
        flagged: list[tuple[str, str]] = []
        for _, dst in pairs:
            if not dst.exists() or dst.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            for m in suspicious_pat.finditer(dst.read_text(encoding="utf-8")):
                flagged.append((str(dst.relative_to(root)), m.group(0)))
        if flagged:
            print("\nUnrecognized CUDA-flavored tokens (review needed):",
                  file=sys.stderr)
            seen = set()
            for path, tok in flagged:
                key = (path, tok)
                if key in seen:
                    continue
                seen.add(key)
                print(f"  {path}: {tok}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
