set(TARGET_NAME ${CMAKE_PROJECT_NAME}_demo)

file(GLOB_RECURSE demo_sources *.cpp)
file(GLOB_RECURSE demo_headers *.h)
add_executable(${CMAKE_PROJECT_NAME}_demo ${demo_sources} ${demo_headers})

target_link_libraries(${TARGET_NAME} ${CMAKE_PROJECT_NAME}_lib)


