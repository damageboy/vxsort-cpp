set(TARGET_NAME ${CMAKE_PROJECT_NAME}_lib)

file(GLOB_RECURSE lib_SRC *.cpp)
file(GLOB_RECURSE lib_HEADERS *.h)
add_library(${TARGET_NAME} STATIC ${lib_SRC} ${lib_HEADERS})
set_target_properties(${TARGET_NAME} PROPERTIES PREFIX lib)
set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${CMAKE_PROJECT_NAME})
target_include_directories(${TARGET_NAME} PUBLIC .)
#set_property(TARGET ${TARGET_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)

