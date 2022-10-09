#include "compiler.h"

#if defined(VXSORT_COMPILER_CLANG) || defined(VXSORT_COMPILER_CLANGCL)
#define VXSORT_TARGET_PUSHED 1
#pragma clang attribute push (__attribute__((target("avx2,popcnt"))), apply_to = any(function))
#endif

#if defined(VXSORT_COMPILER_GCC)
#define VXSORT_TARGET_PUSHED 1
#pragma GCC push_options
#pragma GCC target("avx2")
#endif
