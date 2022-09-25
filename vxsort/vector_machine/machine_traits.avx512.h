
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
using namespace vxsort::types;

#include "avx512/f64.h"
#include "avx512/f32.h"
#include "avx512/i16.h"
#include "avx512/i32.h"
#include "avx512/i64.h"
#include "avx512/u16.h"
#include "avx512/u32.h"
#include "avx512/u64.h"
}

#include "vxsort_targets_disable.h"
#endif  // VXSORT_VXSORT_AVX512_H
