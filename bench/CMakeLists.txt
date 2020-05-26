set(TARGET_NAME ${CMAKE_PROJECT_NAME}_bench)


find_package(Threads REQUIRED)

file(GLOB bench_sources BM_sort.cpp bench.cpp)
file(GLOB bench_headers *.h)
add_executable(${CMAKE_PROJECT_NAME}_bench ${bench_sources} ${bench_headers})

target_link_libraries(${TARGET_NAME}
        gcsort_lib
        benchmark
        ${CMAKE_THREAD_LIBS_INIT})

configure_file(run.sh run.sh COPYONLY)

configure_file(run.cmd run.cmd COPYONLY)

add_test(${TARGET_NAME} gcsort_bench)

