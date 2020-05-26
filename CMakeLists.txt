cmake_minimum_required(VERSION 3.10)

project(gcsort)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)

# Set a default build type if none was specified
set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui

    set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING
            "Available build-types: Debug, Release")


  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "RelWithDebInfo")
endif()

if (CLR_CMAKE_HOST_WIN32)
  message(STATUS "VS_PLATFORM_TOOLSET is ${CMAKE_VS_PLATFORM_TOOLSET}")
  message(STATUS "VS_PLATFORM_NAME is ${CMAKE_VS_PLATFORM_NAME}")
endif (CLR_CMAKE_HOST_WIN32)

if(MSVC)
    add_compile_options(/Zi /FC /Zc:strictStrings /arch:AVX2)

    set(CMAKE_CXX_FLAGS_RELEASE "/O2b2y- -DNDEBUG")

    add_link_options($<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:/DEBUG>)
    add_link_options($<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:/STACK:1572864>)

      # Release build specific flags
      add_link_options($<$<CONFIG:RELEASE>:/OPT:REF>)
      add_link_options($<$<CONFIG:RELEASE>:/OPT:ICF>)
      if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
          add_link_options($<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:/PDBCOMPRESS>)
        set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS_RELEASE} /LTCG")
        add_link_options($<$<CONFIG:RELEASE>:/LTCG>)
      endif()
else()
    add_compile_options(-g3)
    add_compile_options(-Wall)
    add_compile_options(-mavx2)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-Wno-null-conversion)
    else()
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror=conversion-null>)
    endif()

    set(SANITIZERS "-fsanitize=address,bounds,alignment,bool,enum,shift,shift-exponent")

    # c++11 is required
    #
    #set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -DBENCHMARK_ENABLE_LTO=true")
    set(CMAKE_CXX_FLAGS_DEBUG   "${SANITIZERS}")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
endif()



# Location of additional cmake modules
#
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake)

# Guard against in-source builds and bad build-type strings
#
include(ConfigSafeGuards)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# If you want to self-test benchmark lib too, turn me ON
#
set(BENCHMARK_ENABLE_TESTING OFF)

# Configure google micro benchmark
#


include(ConfigGBench)
include(ConfigGTest)

add_subdirectory(${PROJECT_SOURCE_DIR}/gcsort/)
add_subdirectory(${PROJECT_SOURCE_DIR}/bench/)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests/)
