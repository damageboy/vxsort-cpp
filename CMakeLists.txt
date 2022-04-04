cmake_minimum_required(VERSION 3.15)

project(vxsort)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VXSORT_USE_SANITIZERS "Turn on sanitizers" OFF)
option(VXSORT_USE_LIBCXX     "Build and test with libc++ stdlib. (Linux/macOS)" OFF)
option(VXSORT_ENABLE_LTO     "Build and test with link-time optimization"       OFF)
option(VXSORT_GENERATE_STATS "Build auxiliary statistics collection code"       OFF)
option(VXSORT_CCACHE         "Use ccache to speed up re-compilation"            OFF)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM AND NOT ${DONT_USE_CCACHE})
    message("ccache detected - using ccache to cache object files across compilations")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

# Make sure we can import out CMake functions
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CheckCXXCompilerFlag)
include(GetHostType)
include(CPM)

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

function(add_global_cxx_flag FLAG)
    if(ARGC GREATER 1)
        set(VARIANT ${ARGV1})
        string(TOUPPER "_${VARIANT}" VARIANT)
    else()
        set(VARIANT "")
    endif()
    set(CMAKE_CXX_FLAGS${VARIANT} "${CMAKE_CXX_FLAGS${VARIANT}} ${FLAG}" PARENT_SCOPE)
endfunction()

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

    check_cxx_compiler_flag(-Walign-mismatch HAVE_ALIGN_MISMATCH)

    if (HAVE_ALIGN_MISMATCH)
        add_compile_options(-Wno-align-mismatch)
    endif()

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-Wno-null-conversion)
    else()
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror=conversion-null>)
    endif()

    if (${VXSORT_USE_SANITIZERS})
        add_compile_options(-fsanitize=address,bounds,alignment,shift,shift-exponent)
        add_link_options(-fsanitize=address,bounds,alignment,shift,shift-exponent)
    endif()

    if (VXSORT_USE_LIBCXX)
        if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
            add_global_cxx_flag(-stdlib=libc++)
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
                "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            add_global_cxx__flag(-nostdinc++)
            message(WARNING "libc++ header path must be manually specified using CMAKE_CXX_FLAGS")
        else()
            message(FATAL_ERROR "-DVXSORT_USE_LIBCXX:BOOL=ON is not supported for compiler")
        endif()
    endif(VXSORT_USE_LIBCXX)

    if (VXSORT_ENABLE_LTO)
          # Link time optimisation
          add_global_cxx_flag(-flto)
          if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
              find_program(GCC_AR gcc-ar)
              if (GCC_AR)
                  set(CMAKE_AR ${GCC_AR})
              endif()
              find_program(GCC_RANLIB gcc-ranlib)
              if (GCC_RANLIB)
                  set(CMAKE_RANLIB ${GCC_RANLIB})
              endif()
          elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
               find_package(LLVMAr REQUIRED)
               set(CMAKE_AR "${LLVMAR_EXECUTABLE}" CACHE FILEPATH "" FORCE)
               find_package(LLVMNm REQUIRED)
               set(CMAKE_NM "${LLVMNM_EXECUTABLE}" CACHE FILEPATH "" FORCE)
               find_package(LLVMRanLib REQUIRED)
               set(CMAKE_RANLIB "${LLVMRANLIB_EXECUTABLE}" CACHE FILEPATH "" FORCE)
          endif()
    endif()

    if (VXSORT_GENERATE_STATS)
        add_compile_definitions(VXSORT_STATS)
    endif()

    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -fomit-frame-pointer -DNDEBUG -fsave-optimization-record")
endif()

# Guard against in-source builds and bad build-type strings
#
include(ConfigSafeGuards)

#################################################################3
# 3rd party deps
#

CPMAddPackage("gh:google/googletest#main")
CPMAddPackage("gh:fmtlib/fmt#8.1.1")
CPMAddPackage("gh:google/cpu_features#main")

add_subdirectory(${PROJECT_SOURCE_DIR}/vxsort/)
add_subdirectory(${PROJECT_SOURCE_DIR}/demo/)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests/)

if (${CMAKE_SYSTEM_NAME} MATCHES "Android")
    MESSAGE("Skipping google-bench related project since building for Android")
else()
    CPMAddPackage(
      NAME benchmark
      GITHUB_REPOSITORY google/benchmark
      GIT_TAG main
      OPTIONS "BENCHMARK_ENABLE_TESTING OFF" "BENCHMARK_ENABLE_LIBPFM ON" "BENCHMARK_ENABLE_LTO ${VXSORT_ENABLE_LTO}" "BENCHMARK_USE_LIBCXX ${VXSORT_USE_LIBCXX}"
    )

    add_subdirectory(${PROJECT_SOURCE_DIR}/bench/)
endif()
