set(TARGET_NAME ${CMAKE_PROJECT_NAME}_bench)


find_package(Threads REQUIRED)

file(GLOB_RECURSE bench_sources *.cpp)
file(GLOB_RECURSE bench_headers *.h)
add_executable(${CMAKE_PROJECT_NAME}_bench ${bench_sources} ${bench_headers})

target_link_libraries(${TARGET_NAME}
        ${CMAKE_PROJECT_NAME}_lib
        benchmark
        picosha2
        ${CMAKE_THREAD_LIBS_INIT})

configure_file(run.sh run.sh COPYONLY)

configure_file(run.cmd run.cmd COPYONLY)

add_test(${TARGET_NAME} ${TARGET_NAME})

