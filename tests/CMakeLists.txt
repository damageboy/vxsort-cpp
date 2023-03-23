set(TARGET_NAME ${CMAKE_PROJECT_NAME}_tests)

set(test_HEADERS
        smallsort/smallsort_test.h
        fullsort/fullsort_test.h
        mini_tests/pack_machine_test.h
        mini_tests/partition_machine_test.h
        mini_tests/masked_load_store_test.h
)

list(APPEND i_sort_types
        i16
        i32
        i64
)

list(APPEND u_sort_types
        u16
        u32
        u64
)

list(APPEND f_sort_types
        f32
        f64
)

list(APPEND sort_types
        i
        u
        f
)

list(APPEND x86_isas
        avx2
        avx512
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



    foreach(v ${x86_isas})
        foreach(tf ${sort_types})
        string(TOUPPER ${v} vu)
        add_executable(${TARGET_NAME}_${v}_${tf} ${test_${v}_SOURCES} ${test_HEADERS})

        foreach(t ${${tf}_sort_types})
            string(TOUPPER ${t} tu)
            target_compile_definitions(${TARGET_NAME}_${v}_${tf} PRIVATE VXSORT_TEST_${vu}_${tu})
        endforeach ()

        target_link_libraries(${TARGET_NAME}_${v}_${tf}
            ${CMAKE_PROJECT_NAME}_lib
                GTest::gtest_main
                )

        #gtest_discover_tests(${TARGET_NAME}_avx2)
        endforeach()
    endforeach()

endif()

