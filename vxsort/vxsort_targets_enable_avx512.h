#include "compiler.h"

#if defined(VXSORT_COMPILER_CLANG) || defined(VXSORT_COMPILER_CLANGCL)
#define VXSORT_TARGET_PUSHED 1
#pragma clang attribute push (__attribute__((target("avx512f,avx512dq,avx512bw,avx512vbmi2,popcnt"))), apply_to = any(function))
#endif

#if defined(VXSORT_COMPILER_GCC)
#define VXSORT_TARGET_PUSHED 1
#pragma GCC push_options
#pragma GCC target("avx512f,avx512dq,avx512bw,avx512vbmi2,popcnt")
#endif
