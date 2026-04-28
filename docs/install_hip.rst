.. _install_hip:

Installation (AMD/HIP backend, libhipfinufft)
=============================================

This page covers building the **AMD/HIP backend** of FINUFFT
(``libhipfinufft``). For the NVIDIA/CUDA backend see :ref:`install_gpu`,
and for the CPU library see :ref:`install`.

The HIP backend is a sibling of ``libcufinufft`` -- both libraries can be
built and installed in the same tree without conflicting. The two expose
distinct C symbols (``cufinufft_*`` vs ``hipfinufft_*``) so an application
that wants to dispatch between them at runtime can link both.


Requirements
------------

* CMake **>= 3.21** (3.24 recommended; required for the modern ``HIP``
  language support that this build uses).
* ROCm **>= 5.4** (earlier versions may work but are untested; some HIP
  features used here -- memory pools, ``hipMallocAsync`` -- were stabilized
  in 5.4).
* The ROCm packages: ``hip-runtime-amd``, ``hipfft``, ``rocthrust``,
  ``rocprim`` (often pulled in transitively by ``rocthrust``).
* A supported AMD GPU. The build tries to autodetect the local ``gfx``
  architecture via ``rocminfo``; if that fails it falls back to ``native``.


Quick start
-----------

From the repository root::

    cmake -S . -B build-hip \
        -DFINUFFT_USE_CPU=OFF \
        -DFINUFFT_USE_HIP=ON \
        -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc \
        -DROCM_PATH=/opt/rocm
    cmake --build build-hip -j

That produces ``build-hip/libhipfinufft.so`` (or ``.a`` if you set
``-DFINUFFT_STATIC_LINKING=ON``).

To also build the test suite and examples::

    cmake -S . -B build-hip \
        -DFINUFFT_USE_CPU=OFF \
        -DFINUFFT_USE_HIP=ON \
        -DFINUFFT_BUILD_TESTS=ON \
        -DFINUFFT_BUILD_EXAMPLES=ON \
        -DBUILD_TESTING=ON
    cmake --build build-hip -j
    ctest --test-dir build-hip --output-on-failure


Building both backends together
-------------------------------

The two GPU options are independent. To build CUDA and HIP side by side::

    cmake -S . -B build-both \
        -DFINUFFT_USE_CPU=OFF \
        -DFINUFFT_USE_CUDA=ON \
        -DFINUFFT_USE_HIP=ON
    cmake --build build-both -j

The resulting tree contains ``libcufinufft.{a,so}`` and
``libhipfinufft.{a,so}`` and exposes both ``CUFINUFFT_*`` and
``HIPFINUFFT_*`` ctest targets.


Build options
-------------

All FINUFFT-level options apply (see :ref:`install`); the HIP-specific ones
are:

``FINUFFT_USE_HIP`` (default OFF)
    Build the HIP backend.

``CMAKE_HIP_ARCHITECTURES`` (default: autodetect)
    Semicolon-separated list of GPU architectures, e.g.
    ``-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx942"``. If unset, ``cmake/hip_setup.cmake``
    runs ``rocminfo`` to detect what's installed and uses that. If
    ``rocminfo`` is unavailable it falls back to ``native``.

``ROCM_PATH`` (default: ``$ROCM_PATH`` or ``/opt/rocm``)
    Where to find the ROCm toolchain and CMake config files for ``hip``,
    ``hipfft``, and ``rocthrust``.


Public C API
------------

The HIP backend mirrors ``cufinufft``'s public surface. The headers are
``<hipfinufft.h>`` and ``<hipfinufft_opts.h>``; the symbols are::

    void hipfinufft_default_opts(hipfinufft_opts *);

    int hipfinufft_makeplan (int type, int dim, const int64_t *n_modes,
                             int iflag, int ntr, double eps,
                             hipfinufft_plan *plan, const hipfinufft_opts *);
    int hipfinufftf_makeplan(...);  /* float variant */

    int hipfinufft_setpts (hipfinufft_plan, int64_t M, const double *x,
                           const double *y, const double *z, int N,
                           const double *s, const double *t, const double *u);
    int hipfinufftf_setpts(...);

    int hipfinufft_execute (hipfinufft_plan,  hipDoubleComplex *c,
                            hipDoubleComplex *fk);
    int hipfinufftf_execute(hipfinufftf_plan, hipFloatComplex  *c,
                            hipFloatComplex  *fk);

    int hipfinufft_destroy (hipfinufft_plan);
    int hipfinufftf_destroy(hipfinufftf_plan);

The semantics, options struct layout, and supported transform types
(1D/2D/3D, type 1/2/3, single- and double-precision) match cufinufft
exactly.


Migrating from cufinufft
------------------------

In most cases a search-and-replace is sufficient::

    cufinufft   -> hipfinufft
    cufinufftf  -> hipfinufftf
    cuFloatComplex   -> hipFloatComplex
    cuDoubleComplex  -> hipDoubleComplex
    cudaMalloc/cudaMemcpy/cudaFree  -> hipMalloc/hipMemcpy/hipFree
    #include <cufinufft.h>  -> #include <hipfinufft.h>
    #include <cuComplex.h>  -> #include <hip/hip_complex.h>


Known limitations
-----------------

The first cut of this port has the following caveats. Most are tagged in
the source with ``TODO(hip):``.

* **Performance is not yet tuned for AMD.** The bin-size and ``np`` defaults
  in ``hipfinufft_default_opts`` were chosen for NVIDIA SMs (32-thread
  warps). AMD wavefronts are 32 (RDNA) or 64 (CDNA), and the optimal
  shared-memory tile sizes differ accordingly. Numerical results should
  match the CUDA backend within the same tolerance, but throughput will
  often be lower than on a similarly-priced NVIDIA part until the defaults
  are re-tuned.

* **Cache-hint intrinsics fall back to plain loads.** ``intrinsics.h``
  uses ``__ldg``, ``__ldca``, ``__ldcg``, ``__ldcs``, ``__ldcv``, ``__ldlu``
  on NVIDIA. On HIP the corresponding ``#else`` branches resolve to plain
  ``*ptr`` loads. ``__ldg`` is in fact supported on HIP and could be re-
  enabled for a small read-only-cache speed-up; the others have no AMD
  equivalent and stay as plain loads.

* **Shared-memory bank-width hint stripped.**
  ``cudaFuncSetSharedMemConfig(kernel, cudaSharedMemBankSizeEightByte)``
  is NVIDIA-specific. The translator drops the call. The surrounding
  ``THROW_IF_HIP_ERROR`` is preserved (it also checks the prior
  ``hipfinufft_set_shared_memory`` call).

* **Memory-pool allocator (``hipMallocAsync``).** ``ThrustAllocatorAsync``
  in ``hipfinufft_plan_t.h`` queries
  ``hipDeviceAttributeMemoryPoolsSupported`` at runtime. On older ROCm
  (< 5.2) this attribute is unsupported and the allocator falls back to
  plain ``hipMalloc/hipFree``, exactly mirroring the CUDA fallback.

* **Out-of-scope wrappers.** Python (``python/cufinufft``), MATLAB
  (``matlab/cufinufft.{cu,mw}``), Fortran, and Julia bindings are NOT
  ported in this initial drop. They have their own build/codegen flows
  and would need separate translator passes.


Re-syncing the HIP tree
-----------------------

The HIP source tree under ``src/hip/``, ``include/hipfinufft/``,
``test/hip/``, ``examples/hip/``, and ``perftest/hip/`` is auto-generated
from the CUDA tree by ``tools/hipfinufft/cuda_to_hip.py``. When upstream
CUDA code changes, regenerate with::

    python3 tools/hipfinufft/cuda_to_hip.py

To verify the HIP tree is in sync (CI use)::

    python3 tools/hipfinufft/cuda_to_hip.py --check

To audit residual CUDA-flavored tokens after translation (post-rewrite
sanity check)::

    python3 tools/hipfinufft/cuda_to_hip.py --report

The four ``CMakeLists.txt`` files in the HIP tree are *hand-written*, not
generated; the translator skips them. If you add new HIP-specific build
logic, edit those files directly.
