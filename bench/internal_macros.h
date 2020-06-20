#ifndef VXSORT_BENC_INTERNAL_MACROS_H_
#define VXSORT_BENC_INTERNAL_MACROS_H_

#include "benchmark/benchmark.h"

/* Needed to detect STL */
#include <cstdlib>

// clang-format off

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if defined(__clang__)
  #if !defined(COMPILER_CLANG)
    #define COMPILER_CLANG
  #endif
#elif defined(_MSC_VER)
  #if !defined(COMPILER_MSVC)
    #define COMPILER_MSVC
  #endif
#elif defined(__GNUC__)
  #if !defined(COMPILER_GCC)
    #define COMPILER_GCC
  #endif
#endif

#if __has_feature(cxx_attributes)
  #define VXSORT_BENCH_NORETURN [[noreturn]]
#elif defined(__GNUC__)
  #define VXSORT_BENCH_NORETURN __attribute__((noreturn))
#elif defined(COMPILER_MSVC)
  #define VXSORT_BENCH_NORETURN __declspec(noreturn)
#else
  #define VXSORT_BENCH_NORETURN
#endif

#if defined(__CYGWIN__)
  #define VXSORT_BENCH_OS_CYGWIN 1
#elif defined(_WIN32)
  #define VXSORT_BENCH_OS_WINDOWS 1
  #if defined(__MINGW32__)
    #define VXSORT_BENCH_OS_MINGW 1
  #endif
#elif defined(__APPLE__)
  #define VXSORT_BENCH_OS_APPLE 1
  #include "TargetConditionals.h"
  #if defined(TARGET_OS_MAC)
    #define VXSORT_BENCH_OS_MACOSX 1
    #if defined(TARGET_OS_IPHONE)
      #define VXSORT_BENCH_OS_IOS 1
    #endif
  #endif
#elif defined(__FreeBSD__)
  #define VXSORT_BENCH_OS_FREEBSD 1
#elif defined(__NetBSD__)
  #define VXSORT_BENCH_OS_NETBSD 1
#elif defined(__OpenBSD__)
  #define VXSORT_BENCH_OS_OPENBSD 1
#elif defined(__linux__)
  #define VXSORT_BENCH_OS_LINUX 1
#elif defined(__native_client__)
  #define VXSORT_BENCH_OS_NACL 1
#elif defined(__EMSCRIPTEN__)
  #define VXSORT_BENCH_OS_EMSCRIPTEN 1
#elif defined(__rtems__)
  #define VXSORT_BENCH_OS_RTEMS 1
#elif defined(__Fuchsia__)
#define VXSORT_BENCH_OS_FUCHSIA 1
#elif defined (__SVR4) && defined (__sun)
#define VXSORT_BENCH_OS_SOLARIS 1
#elif defined(__QNX__)
#define VXSORT_BENCH_OS_QNX 1
#endif

#if defined(__ANDROID__) && defined(__GLIBCXX__)
#define VXSORT_BENCH_STL_ANDROID_GNUSTL 1
#endif

#if !__has_feature(cxx_exceptions) && !defined(__cpp_exceptions) \
     && !defined(__EXCEPTIONS)
  #define VXSORT_BENCH_HAS_NO_EXCEPTIONS
#endif

#if defined(COMPILER_CLANG) || defined(COMPILER_GCC)
  #define VXSORT_BENCH_MAYBE_UNUSED __attribute__((unused))
#else
  #define VXSORT_BENCH_MAYBE_UNUSED
#endif

// clang-format on

#endif  // BENCHMARK_INTERNAL_MACROS_H_
