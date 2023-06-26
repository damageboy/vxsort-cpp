cmake_minimum_required(VERSION 3.24)

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
option(VXSORT_TIMETRACE      "Produce Clang Time-Trace .json files"             OFF)
set(VXSORT_USE_LINKER "" CACHE STRING "Custom linker for -fuse-ld=...")

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM AND ${VXSORT_CCACHE})
    message("ccache detected - using ccache to cache object files across compilations")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

# Make sure we can import out CMake functions
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CheckCXXCompilerFlag)
include(GetHostType)
include(CPM)
include(CheckLinkerFlag)

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
    add_compile_options(/FC /Zc:strictStrings /EHs)
    add_compile_options(
    "$<$<CONFIG:DEBUG>:/Zi>"
    "$<$<CONFIG:Release>:/Z7>")

    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    set(CMAKE_CXX_FLAGS_RELEASE "/O2b2y- -DNDEBUG")

    add_link_options($<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:/DEBUG>)
    add_link_options($<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:/STACK:1572864>)

    # Release build specific flags
    add_link_options($<$<CONFIG:RELEASE>:/OPT:REF>)
    add_link_options($<$<CONFIG:RELEASE>:/OPT:ICF>)
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
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

    if(${VXSORT_TIMETRACE})
      if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-ftime-trace)
      else()
        message(ERROR "Only clang supports time-tracing")
      endif()
    endif()

    function(get_linker_candidates result)
        set(candidates "")

        if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
            string(REPLACE "." ";" clang_version_components ${CMAKE_C_COMPILER_VERSION})
            list(GET clang_version_components 0 clang_major)
            list(APPEND candidates lld-${clang_major} lld)
        endif ()

        set(${result} ${candidates} PARENT_SCOPE)
    endfunction()

    # Hack/best-effort: we need this for LTO/IPO, which is unconditionally enabled
    # for Release/RelWithDebInfo below. System-default clang will work, but
    # non-system-default clang may select the system-default lld when run through
    # various symlinks. This will select the wrong lld version and break LTO.
    if (NOT VXSORT_USE_LINKER)
        get_linker_candidates(_candidates)
        foreach (candidate ${_candidates})
            set(varname "HAS_${candidate}")
            string(REPLACE "-" "_" varname ${varname})
            check_linker_flag(CXX "-fuse-ld=${candidate}" ${varname})
            if (NOT ${varname})
                continue()
            endif ()

            set(VXSORT_USE_LINKER ${candidate} CACHE STRING "Custom linker for -fuse-ld=..." FORCE)
            break()
        endforeach ()
    endif ()

    if (VXSORT_USE_LINKER)
        add_link_options(-fuse-ld=${VXSORT_USE_LINKER})
        message(STATUS "Overriding linker to ${VXSORT_USE_LINKER}")
    endif ()

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

    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -fomit-frame-pointer -DNDEBUG")
endif()

# Guard against in-source builds and bad build-type strings
#
include(ConfigSafeGuards)

#################################################################3
# 3rd party deps
#

CPMAddPackage(
        NAME backward-cpp
        GITHUB_REPOSITORY bombela/backward-cpp
        GIT_TAG master
)

CPMAddPackage(
        NAME googletest
        GITHUB_REPOSITORY google/googletest
        GIT_TAG v1.13.0
        VERSION 1.13.0
        OPTIONS "BUILD_GMOCK OFF" "INSTALL_GTEST OFF" "gtest_force_shared_crt"
        OVERRIDE_FIND_PACKAGE
)
CPMAddPackage(
        NAME cpu_features
        GITHUB_REPOSITORY google/cpu_features
        GIT_TAG main
        OPTIONS "BUILD_TESTING OFF"
)
CPMAddPackage("gh:fmtlib/fmt#10.0.0")
CPMAddPackage("gh:Neargye/magic_enum#v0.9.2")
CPMAddPackage("gh:okdshin/PicoSHA2#master")

enable_testing()
find_package(GTest)

add_subdirectory(${PROJECT_SOURCE_DIR}/vxsort/)
add_subdirectory(${PROJECT_SOURCE_DIR}/demo/)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests/)

if (${CMAKE_SYSTEM_NAME} MATCHES "Android")
    MESSAGE("Skipping google-bench related project since building for Android")
else()
    CPMAddPackage(
      NAME benchmark
      GITHUB_REPOSITORY damageboy/benchmark
      GIT_TAG main
      OPTIONS "BENCHMARK_ENABLE_TESTING OFF" "BENCHMARK_ENABLE_LIBPFM ON" "BENCHMARK_ENABLE_LTO ${VXSORT_ENABLE_LTO}" "BENCHMARK_USE_LIBCXX ${VXSORT_USE_LIBCXX}"
    )

    add_subdirectory(${PROJECT_SOURCE_DIR}/bench/)
endif()
