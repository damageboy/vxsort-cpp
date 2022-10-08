#include "compiler.h"

#ifdef VXSORT_TARGET_PUSHED

#if defined(VXSORT_COMPILER_CLANG) || defined(VXSORT_COMPILER_CLANGCL)
#pragma clang attribute pop
#endif

#if defined(VXSORT_COMPILER_GCC)
#pragma GCC pop_options
#endif


#endif
