#ifndef VXSORT_COMPILER_H
#define VXSORT_COMPILER_H

#ifdef _MSC_VER
#ifdef __clang__
#define VXSORT_COMPILER_CLANGCL _MSC_VER
#else // real MSVC
#define VXSORT_COMPILER_MSVC _MSC_VER
#endif
#else
#ifdef __GNUC__
#ifdef __clang__
#define VXSORT_COMPILER_CLANG (__clang_major__ * 100 + __clang_minor__)
#else
#define VXSORT_COMPILER_GCC (__GNUC__ * 100 + __GNUC_MINOR__)
#endif
#endif
#endif

#endif  // VXSORT_COMPILER_H
