set(TARGET_NAME ${CMAKE_PROJECT_NAME}_demo)

file(GLOB_RECURSE demo_sources *.cpp)
add_executable(${CMAKE_PROJECT_NAME}_demo ${demo_sources})

target_link_libraries(${TARGET_NAME} ${CMAKE_PROJECT_NAME}_lib)
