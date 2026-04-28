# test/hip/CMakeLists.txt
#
# Mirrors test/cuda/CMakeLists.txt one-for-one. Test names are renamed
# cufinufft* -> hipfinufft* so ctest can run both backends in a single
# build tree without name clashes.

file(GLOB test_src "*.c" "*.cpp" "*.hip")

foreach(srcfile ${test_src})
    get_filename_component(executable ${srcfile} NAME_WE)
    add_executable(${executable} ${srcfile})

    # Mark .hip files explicitly so the HIP language is picked up regardless
    # of CMake version / project-wide language defaults.
    if(srcfile MATCHES "\\.hip$")
        set_source_files_properties(${srcfile} PROPERTIES LANGUAGE HIP)
        set_target_properties(
            ${executable}
            PROPERTIES
                LINKER_LANGUAGE HIP
                HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
                HIP_STANDARD 17
                HIP_STANDARD_REQUIRED ON
        )
    endif()

    # All transitive dependencies (hip::host, hip::hipfft, roc::rocthrust,
    # finufft_common) come in via hipfinufft's BUILD_INTERFACE.
    target_link_libraries(${executable} PRIVATE hipfinufft)
    if(MATH_LIBRARY)
        target_link_libraries(${executable} PRIVATE ${MATH_LIBRARY})
    endif()
    target_compile_features(${executable} PUBLIC cxx_std_17)
    message(
        STATUS
        "Adding test ${executable}"
        " with HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}"
        " and INCLUDE=${HIPFINUFFT_INCLUDE_DIRS}"
    )
endforeach()

# ---------- ctest entries ---------------------------------------------------
# These mirror test/cuda/CMakeLists.txt exactly: same numerical arguments,
# same coverage of dimensions / methods / precisions / upsamp factors. Only
# the test-name prefix and executable name change.

function(add_tests PREC REQ_TOL CHECK_TOL UPSAMP)
    add_test(
        NAME hipfinufft1d1_test_auto_${PREC}_${UPSAMP}
        COMMAND hipfinufft1d_test 0 1 2e3 4e3 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft1d1_test_GM_${PREC}_${UPSAMP}
        COMMAND hipfinufft1d_test 1 1 2e3 4e3 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft1d1_test_SM_${PREC}_${UPSAMP}
        COMMAND hipfinufft1d_test 2 1 2e3 4e3 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft1d1_test_OD_${PREC}_${UPSAMP}
        COMMAND hipfinufft1d_test 3 1 2e3 4e3 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft1d2_test_auto_${PREC}_${UPSAMP}
        COMMAND hipfinufft1d_test 0 2 2e3 4e3 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft1d2_test_GM_${PREC}_${UPSAMP}
        COMMAND hipfinufft1d_test 1 2 2e3 4e3 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft1d2_test_SM_${PREC}_${UPSAMP}
        COMMAND hipfinufft1d_test 2 2 2e3 4e3 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft1d3_test_auto_${PREC}_${UPSAMP}
        COMMAND hipfinufft1d_test 0 3 2e3 4e3 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft1d3_test_GM_${PREC}_${UPSAMP}
        COMMAND hipfinufft1d_test 1 3 2e3 4e3 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )

    add_test(
        NAME hipfinufft2d1_test_auto_${PREC}_${UPSAMP}
        COMMAND hipfinufft2d_test 0 1 1e2 2e2 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft2d1_test_GM_${PREC}_${UPSAMP}
        COMMAND hipfinufft2d_test 1 1 1e2 2e2 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft2d1_test_SM_${PREC}_${UPSAMP}
        COMMAND hipfinufft2d_test 2 1 1e2 2e2 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft2d1_test_OD_${PREC}_${UPSAMP}
        COMMAND hipfinufft2d_test 3 1 1e2 2e2 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft2d2_test_auto_${PREC}_${UPSAMP}
        COMMAND hipfinufft2d_test 0 2 1e2 2e2 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft2d2_test_GM_${PREC}_${UPSAMP}
        COMMAND hipfinufft2d_test 1 2 1e2 2e2 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft2d2_test_SM_${PREC}_${UPSAMP}
        COMMAND hipfinufft2d_test 2 2 1e2 2e2 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft2d3_test_auto_${PREC}_${UPSAMP}
        COMMAND hipfinufft2d_test 0 3 1e2 2e2 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft2d3_test_SM_${PREC}_${UPSAMP}
        COMMAND hipfinufft2d_test 2 3 1e2 2e2 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft2d3_test_OD_${PREC}_${UPSAMP}
        COMMAND hipfinufft2d_test 3 3 1e2 2e2 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )

    add_test(
        NAME hipfinufft2d1many_test_GM_${PREC}_${UPSAMP}
        COMMAND hipfinufft2dmany_test 1 1 1e2 2e2 5 0 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft2d1many_test_SM_${PREC}_${UPSAMP}
        COMMAND hipfinufft2dmany_test 2 1 1e2 2e2 5 0 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft2d1many_test_OD_${PREC}_${UPSAMP}
        COMMAND hipfinufft2dmany_test 3 1 1e2 2e2 5 0 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft2d2many_test_GM_${PREC}_${UPSAMP}
        COMMAND hipfinufft2dmany_test 1 2 1e2 2e2 5 0 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft2d2many_test_SM_${PREC}_${UPSAMP}
        COMMAND hipfinufft2dmany_test 2 2 1e2 2e2 5 0 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft2d3many_test_GM_${PREC}_${UPSAMP}
        COMMAND hipfinufft2dmany_test 1 3 1e2 2e2 5 0 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft2d3many_test_SM_${PREC}_${UPSAMP}
        COMMAND hipfinufft2dmany_test 2 3 1e2 2e2 5 0 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft2d3many_test_OD_${PREC}_${UPSAMP}
        COMMAND hipfinufft2dmany_test 3 3 1e2 2e2 5 0 2e4 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )

    add_test(
        NAME hipfinufft3d1_test_auto_${PREC}_${UPSAMP}
        COMMAND hipfinufft3d_test 0 1 2 5 10 20 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft3d1_test_GM_${PREC}_${UPSAMP}
        COMMAND hipfinufft3d_test 1 1 2 5 10 20 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )

    # OD method is gated on tolerance like in CUDA path (kept verbatim).
    if(${REQ_TOL} GREATER 1e-11 AND ${UPSAMP} EQUAL 2.0 OR ${REQ_TOL} GREATER 1e-7 AND ${UPSAMP} EQUAL 2.0)
        add_test(
            NAME hipfinufft3d1_test_OD_${PREC}_${UPSAMP}
            COMMAND hipfinufft3d_test 3 1 2 5 10 20 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
        )
    endif()

    if(${PREC} STREQUAL "float")
        add_test(
            NAME hipfinufft3d1_test_SM_${PREC}_${UPSAMP}
            COMMAND hipfinufft3d_test 2 1 2 5 10 20 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
        )
        add_test(
            NAME hipfinufft3d1_test_block_${PREC}_${UPSAMP}
            COMMAND hipfinufft3d_test 4 1 2 5 10 20 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
        )
        add_test(
            NAME hipfinufft3d2_test_SM_${PREC}_${UPSAMP}
            COMMAND hipfinufft3d_test 2 2 2 5 10 20 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
        )
        add_test(
            NAME hipfinufft3d3_test_SM_${PREC}_${UPSAMP}
            COMMAND hipfinufft3d_test 2 3 2 5 10 30 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
        )
    endif()

    add_test(
        NAME hipfinufft3d2_test_auto_${PREC}_${UPSAMP}
        COMMAND hipfinufft3d_test 0 2 2 5 10 20 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft3d2_test_GM_${PREC}_${UPSAMP}
        COMMAND hipfinufft3d_test 1 2 2 5 10 20 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft3d3_test_auto_${PREC}_${UPSAMP}
        COMMAND hipfinufft3d_test 0 3 2 5 10 30 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
    add_test(
        NAME hipfinufft3d3_test_GM_${PREC}_${UPSAMP}
        COMMAND hipfinufft3d_test 1 3 2 3 7 20 ${REQ_TOL} ${CHECK_TOL} ${PREC} ${UPSAMP}
    )
endfunction()

add_test(NAME hipfinufft_public_api      COMMAND public_api_test)
add_test(NAME hipfinufft_makeplan        COMMAND test_makeplan)
add_test(NAME hipfinufft_error_handling  COMMAND hipfinufft_error_handling)
add_test(NAME hipfinufft_math_test       COMMAND hipfinufft_math_test)

add_test(
    NAME hipfinufft1dspreadinterponly_double_test
    COMMAND hipfinufft1dspreadinterponly_test 1e3 1e6 1e-6 1e-5 d 2.00
)
add_test(
    NAME hipfinufft1dspreadinterponly_float_test
    COMMAND hipfinufft1dspreadinterponly_test 1e3 1e6 1e-5 1e-4 f 1.25
)

# Same coverage matrix as CUDA path: 4 (PREC, REQ_TOL, CHECK_TOL, UPSAMP) x 2.
add_tests(float  1e-4  2e-3  2.0)
add_tests(double 1e-12 1e-11 2.0)
add_tests(float  1e-4  2e-3  1.25)
add_tests(double 1e-8  1e-7  1.25)
add_tests(float  1e-4  2e-3  0.f)
add_tests(double 1e-12 1e-11 0.f)
add_tests(float  1e-4  2e-3  0.)
add_tests(double 1e-8  1e-7  0.)
