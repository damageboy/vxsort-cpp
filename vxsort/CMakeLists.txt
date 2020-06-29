set(TARGET_NAME ${CMAKE_PROJECT_NAME}_lib)

set(lib_SRC
        isa_detection_sane.cpp
        stats/vxsort_stats.cpp
)

set(lib_HEADERS
        isa_detection.h
        alignment.h
        defs.h
        packer.h
        vxsort.h
        stats/vxsort_stats.h
        machine_traits.h
        vxsort_targets_disable.h
)

if (${PROCESSOR_IS_X86})
    file(GLOB_RECURSE avx2_SRC       vector_machine/avx2/*.cpp smallsort/avx2/*.cpp)
    file(GLOB_RECURSE avx2_HEADERS   vector_machine/avx2/*.h smallsort/avx2/*.h)
    file(GLOB_RECURSE avx512_SRC     vector_machine/avx512/*.cpp smallsort/avx512/*.cpp)
    file(GLOB_RECURSE avx512_HEADERS vector_machine/avx512/*.h smallsort/avx512/*.h)

    list(APPEND lib_HEADERS
        vector_machine/machine_traits.avx2.h
        vector_machine/machine_traits.avx512.h
        ${avx2_HEADERS}
        ${avx512_HEADERS}
    )

    list(APPEND lib_SRC
            ${avx2_SRC}
            ${avx512_SRC}
    )

endif()

if (${PROCESSOR_IS_ARM} OR ${PROCESSOR_IS_AARCH64})
    file(GLOB_RECURSE neon_SRC vector_machine/neon/*.cpp smallsort/neon/*.cpp)
    file(GLOB_RECURSE neon_HEADERS vector_machine/neon/*.h smallsort/neon/*.h)

    list(APPEND lib_HEADERS ${neon_HEADERS})
    list(APPEND lib_SRC ${neon_SRC})
endif()

add_library(${TARGET_NAME} STATIC ${lib_SRC} ${lib_HEADERS})
set_target_properties(${TARGET_NAME} PROPERTIES PREFIX lib)
set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${CMAKE_PROJECT_NAME})
target_link_libraries(${TARGET_NAME}
    cpu_features
    fmt
)

target_include_directories(${TARGET_NAME} PUBLIC .)
#set_property(TARGET ${TARGET_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)

