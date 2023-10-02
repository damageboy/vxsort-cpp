set(TARGET_NAME ${CMAKE_PROJECT_NAME}_tests)

include(GoogleTest)

set(test_HEADERS
        smallsort/smallsort_test.h
        fullsort/fullsort_test.h
        mini_tests/pack_machine_test.h
        mini_tests/partition_machine_test.h
        mini_tests/masked_load_store_test.h
        test_isa.h)

list(APPEND sort_types
        i
        u
        f
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

list(APPEND x86_isas
        avx2
        avx512
)

list(APPEND test_SOURCES
        gtest_main.cpp
        mini_tests/masked_load_store.sanity.cpp
)

if (${PROCESSOR_IS_X86})
    foreach(v ${x86_isas})
        foreach(tf ${sort_types})
        string(TOUPPER ${v} vu)

        add_executable(${TARGET_NAME}_${v}_${tf} ${test_SOURCES} ${test_HEADERS}
                smallsort/smallsort.${v}.${tf}.cpp
                fullsort/fullsort.${v}.${tf}.cpp
                mini_tests/masked_load_store.${v}.cpp
                mini_tests/partition_machine.${v}.cpp
                mini_tests/pack_machine.${v}.cpp)

        foreach(t ${${tf}_sort_types})
            string(TOUPPER ${tf} tfu)
            string(TOUPPER ${t} tu)
            target_compile_definitions(${TARGET_NAME}_${v}_${tf} PRIVATE VXSORT_TEST_${vu}_${tu} VXSORT_TEST_${vu}_${tfu})
        endforeach ()

        target_link_libraries(${TARGET_NAME}_${v}_${tf}
            ${CMAKE_PROJECT_NAME}_lib
            magic_enum::magic_enum
            Backward::Backward
            GTest::gtest
            )

        add_test(${TARGET_NAME}_${v}_${tf} ${TARGET_NAME}_${v}_${tf})
        endforeach()
    endforeach()

endif()

