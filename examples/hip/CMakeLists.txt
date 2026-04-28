# examples/hip/CMakeLists.txt
#
# Mirrors examples/cuda/CMakeLists.txt. Examples are pure host C++ that talk
# to the hipfinufft public API via hip/* runtime, so we stay in CXX language
# mode and only pull in HIP for the runtime/headers.

file(GLOB example_src "*.cpp")

foreach(srcfile ${example_src})
    get_filename_component(executable ${srcfile} NAME_WE)
    add_executable(${executable} ${srcfile})
    target_include_directories(${executable} PUBLIC ${HIPFINUFFT_INCLUDE_DIRS})
    target_link_libraries(
        ${executable}
        PRIVATE
            hipfinufft
            hip::host
            hip::hipfft
            $<BUILD_INTERFACE:finufft_common>
    )
    target_compile_features(${executable} PRIVATE cxx_std_17)
endforeach()
