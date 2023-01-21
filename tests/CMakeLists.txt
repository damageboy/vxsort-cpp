set(TARGET_NAME ${CMAKE_PROJECT_NAME}_tests)

set(test_HEADERS
        smallsort/smallsort_test.h
        fullsort/fullsort_test.h
        mini_tests/pack_machine_test.h
        mini_tests/partition_machine_test.h
        mini_tests/masked_load_store_test.h
)

list(APPEND test_SOURCES
        mini_tests/masked_load_store.sanity.cpp
)

if (${PROCESSOR_IS_X86})
    set(test_avx2_SOURCES ${test_SOURCES})
    list(APPEND test_avx2_SOURCES
            smallsort/smallsort.avx2.cpp
            fullsort/fullsort.avx2.cpp
            mini_tests/masked_load_store.avx2.cpp
            mini_tests/partition_machine.avx2.cpp
            mini_tests/pack_machine.avx2.cpp
    )

    set(test_avx512_SOURCES ${test_SOURCES})
    list(APPEND test_avx512_SOURCES
            smallsort/smallsort.avx512.cpp
            fullsort/fullsort.avx512.cpp
            mini_tests/masked_load_store.avx512.cpp
            mini_tests/partition_machine.avx512.cpp
            mini_tests/pack_machine.avx512.cpp
            )
endif()

add_executable(${TARGET_NAME}_avx2 ${test_avx2_SOURCES} ${test_HEADERS})

target_link_libraries(${TARGET_NAME}_avx2
        ${CMAKE_PROJECT_NAME}_lib
        gtest
        gmock
        gmock_main)

add_executable(${TARGET_NAME}_avx512 ${test_avx512_SOURCES} ${test_HEADERS})

target_link_libraries(${TARGET_NAME}_avx512
        ${CMAKE_PROJECT_NAME}_lib
        gtest
        gmock
        gmock_main)

gtest_discover_tests(${TARGET_NAME}_avx2)
gtest_discover_tests(${TARGET_NAME}_avx512)
