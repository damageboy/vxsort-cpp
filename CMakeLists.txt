cmake_minimum_required(VERSION 3.15)

project(vxsort)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option (USE_SANITIZERS "Turn on sanitizers" OFF)

# Location of additional cmake modules
#
set(CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH}
        ${PROJECT_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)
include(GetHostType)
include(FetchContent)

# Set a default build type if none was specified
set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
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

    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

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

    if (${USE_SANITIZERS})
        add_compile_options(-fsanitize=address,bounds,alignment,shift,shift-exponent)
        add_link_options(-fsanitize=address,bounds,alignment,shift,shift-exponent)
    endif()

    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -fomit-frame-pointer -DNDEBUG -fsave-optimization-record")
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

# Guard against in-source builds and bad build-type strings
#
include(ConfigSafeGuards)

#################################################################3
# 3rd party deps
#
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG master
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG master
)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(cpu_features
        GIT_REPOSITORY https://github.com/google/cpu_features.git
        GIT_TAG master
)
FetchContent_MakeAvailable(cpu_features)

add_subdirectory(${PROJECT_SOURCE_DIR}/vxsort/)
add_subdirectory(${PROJECT_SOURCE_DIR}/demo/)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests/)

if (${CMAKE_SYSTEM_NAME} MATCHES "Android")
    MESSAGE("Skipping google-bench related project since building for Android")
else()
    set(BENCHMARK_ENABLE_TESTING OFF)
    FetchContent_Declare(googlebench
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG master
    )
    FetchContent_MakeAvailable(googlebench)

    add_subdirectory(${PROJECT_SOURCE_DIR}/bench/)
endif()
