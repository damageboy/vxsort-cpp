set(TARGET_NAME ${CMAKE_PROJECT_NAME}_demo)

set(demo_SOURCES
        demo.cpp)

if (${PROCESSOR_IS_X86})
    list(APPEND demo_SOURCES
            do_avx2.cpp
            do_avx512.cpp)
endif()

add_executable(${CMAKE_PROJECT_NAME}_demo ${demo_SOURCES})

target_link_libraries(${TARGET_NAME} ${CMAKE_PROJECT_NAME}_lib)
