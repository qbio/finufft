# cmake/hip_setup.cmake
#
# Sibling of cmake/cuda_setup.cmake that wires up the HIP toolchain for the
# AMD backend (hipfinufft). The two files do *not* share state; you can build
# either, both, or neither.

include_guard(GLOBAL)

# Detect a sensible default GPU architecture from rocm-info / rocminfo. This
# mirrors what cuda_setup.cmake does with nvidia-smi. Users can override via
# -DCMAKE_HIP_ARCHITECTURES=gfx942;gfx90a or via the CMAKE_HIP_ARCHITECTURES
# preset.
function(detect_hip_architecture)
    find_program(ROCMINFO_EXECUTABLE NAMES rocminfo rocm-info)
    if(ROCMINFO_EXECUTABLE)
        execute_process(
            COMMAND ${ROCMINFO_EXECUTABLE}
            OUTPUT_VARIABLE rocminfo_output
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        # rocminfo lists each GPU's name as "Name:                    gfxNNNN"
        # We capture fully-qualified gfx IDs (4+ hex digits after "gfx").
        # Short aliases like "gfx12" are excluded because clang rejects them.
        string(REGEX MATCHALL "gfx[0-9a-f][0-9a-f][0-9a-f][0-9a-f]+" found_arches "${rocminfo_output}")
        if(found_arches)
            list(REMOVE_DUPLICATES found_arches)
            string(REPLACE ";" " " arch_human "${found_arches}")
            message(STATUS "Detected HIP gfx archs: ${arch_human}")
            set(CMAKE_HIP_ARCHITECTURES "${found_arches}" CACHE STRING "HIP gfx archs" FORCE)
            return()
        endif()
        message(WARNING "Could not parse gfx architectures from rocminfo output. Using 'native'.")
    else()
        message(WARNING "rocminfo / rocm-info not found. Using 'native'.")
    endif()
    set(CMAKE_HIP_ARCHITECTURES "native" CACHE STRING "HIP gfx archs" FORCE)
endfunction()

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
    detect_hip_architecture()
else()
    message(STATUS "Using user-specified CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
endif()

# ROCm install root. The hip / hipfft / rocthrust packages all install their
# CMake config files under <ROCM_PATH>/lib/cmake. We let the user override but
# default to /opt/rocm, which is where the official packages land.
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm install prefix")
    else()
        set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm install prefix")
    endif()
endif()
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}" "${ROCM_PATH}/lib/cmake")
message(STATUS "ROCm install prefix: ${ROCM_PATH}")

# Enable HIP as a first-class language. Requires CMake >= 3.21.
enable_language(HIP)

# Find the runtime libraries we depend on. hipfft is the cuFFT analogue.
# rocthrust ships the thrust:: namespace headers used throughout cufinufft
# (device_vector, device_ptr, raw_pointer_cast, etc.) and is therefore a
# direct stand-in for CCCL/Thrust on the NVIDIA path.
find_package(hip       CONFIG REQUIRED)
find_package(hipfft    CONFIG REQUIRED)
find_package(rocthrust CONFIG REQUIRED)

# Best-effort: some ROCm distributions ship rocprim as a separate package that
# rocthrust depends on transitively. Mark it OPTIONAL so older / packaged
# layouts where it's bundled with rocthrust still work.
find_package(rocprim   CONFIG QUIET)

message(STATUS "Found HIP version:     ${hip_VERSION}")
message(STATUS "Found hipFFT version:  ${hipfft_VERSION}")
message(STATUS "Found rocThrust version: ${rocthrust_VERSION}")

# hipMallocAsync / hipFreeAsync were added in ROCm 5.2. The pool-allocator
# code path in include/hipfinufft/hipfinufft_plan_t.h links these symbols
# unconditionally (the runtime check via hipDeviceAttributeMemoryPoolsSupported
# only chooses between the pool and non-pool entry points; both have to
# resolve). On older ROCm the link will fail with an opaque "undefined
# reference" message; surface a clean error here instead.
if(hip_VERSION VERSION_LESS 5.2)
    message(FATAL_ERROR
        "FINUFFT_USE_HIP requires ROCm >= 5.2 (found hip ${hip_VERSION}). "
        "The libhipfinufft pool-allocator code path uses hipMallocAsync / "
        "hipFreeAsync, which were added in ROCm 5.2. "
        "Upgrade ROCm, or disable HIP support with -DFINUFFT_USE_HIP=OFF.")
endif()
