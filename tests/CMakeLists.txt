set(TARGET_NAME ${CMAKE_PROJECT_NAME}_tests)

#include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

file(GLOB test_SRC *.cpp)
file(GLOB test_HEADERS *.h)

add_executable(${TARGET_NAME} ${test_SRC} ${test_HEADERS})

target_link_libraries(${TARGET_NAME}
        gcsort_lib
        gtest
        gmock
        gmock_main)
