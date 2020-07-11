cmake_minimum_required(VERSION 3.10)

project(vxsort)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 14)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)

# Set a default build type if none was specified
set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui

    set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;Stats" CACHE STRING
            "Available build-types: Debug, Release, RelWithDebInfo, Stats")


  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "RelWithDebInfo" "Stats")
endif()

if (CLR_CMAKE_HOST_WIN32)
  message(STATUS "VS_PLATFORM_TOOLSET is ${CMAKE_VS_PLATFORM_TOOLSET}")
  message(STATUS "VS_PLATFORM_NAME is ${CMAKE_VS_PLATFORM_NAME}")
endif (CLR_CMAKE_HOST_WIN32)

if(MSVC)
    add_compile_options(/Zi /FC /Zc:strictStrings /EHs)

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
    #set(CMAKE_CXX_FLAGS_RELEASE "-march=skylake-avx512 -mtune=skylake-avx512 -O2 -fomit-frame-pointer -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -fomit-frame-pointer -DNDEBUG")
endif()

SET(CMAKE_CXX_FLAGS_STATS
        "${CMAKE_CXX_FLAGS_RELEASE} -DVXSORT_STATS"
        CACHE STRING "Flags used by the C++ compiler during stats builds."
        FORCE )
SET(CMAKE_C_FLAGS_STATS
        "${CMAKE_C_FLAGS_RELEASE} -DVXSORT_STATS"
        CACHE STRING "Flags used by the C++ compiler during stats builds."
        FORCE )
SET(CMAKE_EXE_LINKER_FLAGS_STATS
        "${CMAKE_EXE_LINKER_FLAGS_RELEASE}"
        CACHE STRING "Flags used for linking binaries during coverage builds."
        FORCE )
SET(CMAKE_SHARED_LINKER_FLAGS_STATS
        "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}"
        CACHE STRING "Flags used by the shared libraries linker during coverage builds."
        FORCE )

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

add_subdirectory(${PROJECT_SOURCE_DIR}/vxsort/)
add_subdirectory(${PROJECT_SOURCE_DIR}/bench/)
add_subdirectory(${PROJECT_SOURCE_DIR}/demo/)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests/)
