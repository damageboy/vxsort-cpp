//
// Created by dans on 6/1/20.
//

#ifndef VXSORT_MACHINE_TRAITS_AVX512_H
#define VXSORT_MACHINE_TRAITS_AVX512_H

#include "vxsort_targets_enable_avx512.h"

#include <limits>
#include <immintrin.h>
#include <cassert>
#include <type_traits>
#include "defs.h"
#include "machine_traits.h"



namespace vxsort {
#include "avx512/double.h"
#include "avx512/float.h"
#include "avx512/int32_t.h"
#include "avx512/int64_t.h"
#include "avx512/uint32_t.h"
#include "avx512/uint64_t.h"
}

#include "vxsort_targets_disable.h"

#endif  // VXSORT_VXSORT_AVX512_H
