# src/hip/CMakeLists.txt
#
# Builds libhipfinufft, the AMD/HIP backend. Mirrors src/cuda/CMakeLists.txt
# in scope and behaviour. The two libraries are siblings -- they install
# different file names, expose different C symbols, and can coexist on a
# system. Selection is build-time via FINUFFT_USE_HIP.

include(staticAnalysis)
include(setupIWYU)

set(SRC
    spreadinterp.cpp
    1d/spread1d_wrapper.hip
    1d/interp1d_wrapper.hip
    2d/spread2d_wrapper.hip
    2d/interp2d_wrapper.hip
    3d/spread3d_wrapper.hip
    3d/interp3d_wrapper.hip
    memtransfer_wrapper.hip
    deconvolve_wrapper.hip
    hipfinufft.hip
    common.hip
    hipfinufft_plan_t.hip
)

# Mark every source as HIP so the HIP compiler (clang++) handles all of them.
# spreadinterp.cpp is pure host C++ but clang++ compiles it correctly in HIP
# mode as host-only code. Using the HIP compiler for everything avoids CMake
# injecting -x hip into the g++ CXX_FLAGS for mixed-language targets.
foreach(_src IN LISTS SRC)
    set_source_files_properties(${_src} PROPERTIES LANGUAGE HIP)
endforeach()

set(HIPFINUFFT_INCLUDE_DIRS
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/contrib>
    $<INSTALL_INTERFACE:include>
)
set(HIPFINUFFT_INCLUDE_DIRS ${HIPFINUFFT_INCLUDE_DIRS} PARENT_SCOPE)

# HIP-side compile flags.
# OCML_BASIC_ROUNDED_OPERATIONS enables __fsub_rd/__fmul_rd/__dsub_rd/__dmul_rd
# directed-rounding intrinsics in __clang_hip_math.h (guarded by that macro).
list(APPEND FINUFFT_HIP_FLAGS
    $<$<COMPILE_LANGUAGE:HIP>:-Wextra>
    $<$<COMPILE_LANGUAGE:HIP>:-Wall>
    $<$<COMPILE_LANGUAGE:HIP>:-DOCML_BASIC_ROUNDED_OPERATIONS>
)

if(FINUFFT_SHARED_LINKING)
    add_library(hipfinufft SHARED ${SRC})
else()
    add_library(hipfinufft STATIC ${SRC})
endif()

enable_static_analysis(hipfinufft)
add_iwyu_fix_target(hipfinufft)
set_property(TARGET hipfinufft PROPERTY INTERPROCEDURAL_OPTIMIZATION ${FINUFFT_INTERPROCEDURAL_OPTIMIZATION})
add_library(finufft::hipfinufft ALIAS hipfinufft)

target_include_directories(hipfinufft PUBLIC ${HIPFINUFFT_INCLUDE_DIRS})

set_target_properties(hipfinufft PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

if(FINUFFT_SHARED_LINKING AND FINUFFT_BUILD_TESTS)
    set(FINUFFT_HIP_VISIBILITY_PRESET default)
else()
    set(FINUFFT_HIP_VISIBILITY_PRESET hidden)
endif()

set_target_properties(
    hipfinufft
    PROPERTIES
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
        HIP_SEPARABLE_COMPILATION ON
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        INTERPROCEDURAL_OPTIMIZATION ${FINUFFT_INTERPROCEDURAL_OPTIMIZATION}
        POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE}
        LINKER_LANGUAGE HIP
        CXX_VISIBILITY_PRESET ${FINUFFT_HIP_VISIBILITY_PRESET}
        HIP_VISIBILITY_PRESET ${FINUFFT_HIP_VISIBILITY_PRESET}
        VISIBILITY_INLINES_HIDDEN YES
        WINDOWS_EXPORT_ALL_SYMBOLS NO
)

target_compile_features(hipfinufft PRIVATE cxx_std_17)
target_compile_options(hipfinufft PUBLIC ${FINUFFT_HIP_FLAGS})

if(FINUFFT_SHARED_LINKING)
    target_compile_definitions(hipfinufft PRIVATE FINUFFT_DLL)
    if(WIN32)
        target_compile_definitions(hipfinufft PRIVATE dll_EXPORTS)
    endif()
endif()

# Public/private dependency exposure parallels the CUDA target. Build-side
# consumers (tests, examples) get hip / hipfft / rocthrust on their include
# and link lines automatically. Install-side consumers don't need rocthrust
# re-exposed (it's header-only) and the .so encodes its own runtime deps.
target_link_libraries(
    hipfinufft
    PUBLIC
        $<BUILD_INTERFACE:hip::host>
        $<BUILD_INTERFACE:hip::device>
        $<BUILD_INTERFACE:hip::hipfft>
        $<BUILD_INTERFACE:roc::rocthrust>
        $<BUILD_INTERFACE:finufft_common>
)

# Some hipcc + ROCm versions warn on deprecated thrust APIs. Mute that since
# the warning is about transitions outside our control.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-Wno-deprecated-declarations FINUFFT_HIP_HAS_NO_DEPRECATED_DECLARATIONS)
if(FINUFFT_HIP_HAS_NO_DEPRECATED_DECLARATIONS)
    target_compile_options(hipfinufft PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-Wno-deprecated-declarations>)
endif()

file(GLOB HIPFINUFFT_PUBLIC_HEADERS "${CMAKE_SOURCE_DIR}/include/hipfinufft*.h")
set_target_properties(hipfinufft PROPERTIES PUBLIC_HEADER "${HIPFINUFFT_PUBLIC_HEADERS}")
