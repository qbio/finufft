# perftest/hip/CMakeLists.txt
#
# Mirrors perftest/cuda/CMakeLists.txt. Two perf executables, both built as
# HIP-language sources.

add_executable(hipperftest hipperftest.hip)
set_source_files_properties(hipperftest.hip PROPERTIES LANGUAGE HIP)
target_link_libraries(hipperftest PRIVATE hipfinufft)
target_compile_features(hipperftest PRIVATE cxx_std_17)
set_target_properties(
    hipperftest
    PROPERTIES
        LINKER_LANGUAGE HIP
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
        HIP_STANDARD 17
        HIP_STANDARD_REQUIRED ON
)

add_executable(binsize_sweep_hip binsize_sweep.hip)
set_source_files_properties(binsize_sweep.hip PROPERTIES LANGUAGE HIP)
target_link_libraries(binsize_sweep_hip PRIVATE hipfinufft)
target_compile_features(binsize_sweep_hip PRIVATE cxx_std_17)
set_target_properties(
    binsize_sweep_hip
    PROPERTIES
        LINKER_LANGUAGE HIP
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
        HIP_STANDARD 17
        HIP_STANDARD_REQUIRED ON
)
