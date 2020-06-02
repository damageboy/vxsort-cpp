#ifndef GCSORT_VXSORT_INT64_T_H
#define GCSORT_VXSORT_INT64_T_H

#include "defs.h"
#include "vxsort.h"

#ifdef ARCH_X64
#include "machine_traits.avx2.h"
#include "machine_traits.avx512.h"
#include "smallsort/bitonic_sort.AVX2.int64_t.generated.h"
#include "smallsort/bitonic_sort.AVX512.int64_t.generated.h"
#endif
#endif  // GCSORT_VXSORT_INT32_T_H
