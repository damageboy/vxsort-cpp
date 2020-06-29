set(TARGET_NAME ${CMAKE_PROJECT_NAME}_tests)

#include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

set(test_HEADERS
        smallsort/smallsort_test.h
        fullsort/fullsort_test.h
        packunpack/packunpack_test.h
)

add_compile_definitions("BITONIC_TESTS")

if (${PROCESSOR_IS_X86})
    list(APPEND test_SOURCES
            smallsort/smallsort.avx2.cpp
            smallsort/smallsort.avx512.cpp
            fullsort/fullsort.avx2.cpp
            fullsort/fullsort.avx512.cpp
            packunpack/packunpack.avx2.cpp
            packunpack/packunpack.avx512.cpp
    )
endif()

if (${PROCESSOR_IS_ARM} OR ${PROCESSOR_IS_AARCH64})
    list(APPEND test_SOURCES
            smallsort/smallsort.neon.cpp
    )
endif()


add_executable(${TARGET_NAME} ${test_SOURCES} ${test_HEADERS})

target_link_libraries(${TARGET_NAME}
        ${CMAKE_PROJECT_NAME}_lib
        gtest
        gmock
        gmock_main)
