set(TARGET_NAME ${CMAKE_PROJECT_NAME}_tests)

#include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

set(test_HEADERS
        smallsort/smallsort_test.h
        fullsort/fullsort_test.h
)

add_compile_definitions("BITONIC_TESTS")

list(APPEND test_SOURCES
        mini_tests/masked_load_store.sanity.cpp
)

if (${PROCESSOR_IS_X86})
    list(APPEND test_SOURCES
            smallsort/smallsort.avx2.cpp
            smallsort/smallsort.avx512.cpp
            fullsort/fullsort.avx2.cpp
            fullsort/fullsort.avx512.cpp
            mini_tests/masked_load_store.avx2.cpp
            mini_tests/masked_load_store.avx512.cpp
            mini_tests/partition_machine.avx2.cpp
            mini_tests/partition_machine.avx512.cpp
    )
endif()

add_executable(${TARGET_NAME} ${test_SOURCES} ${test_HEADERS})

target_link_libraries(${TARGET_NAME}
        ${CMAKE_PROJECT_NAME}_lib
        gtest
        gmock
        gmock_main)
