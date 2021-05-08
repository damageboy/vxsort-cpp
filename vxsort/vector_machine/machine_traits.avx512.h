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

struct __m512_dbg {
    using TV = __m512i;
    static const int V_SIZE = sizeof(TV);
   public:
    union {
        alignas(V_SIZE) i8      i8[V_SIZE/sizeof(i8)];
        alignas(V_SIZE) i16     i16[V_SIZE/sizeof(i16)];
        alignas(V_SIZE) i32     i32[V_SIZE/sizeof(i32)];
        alignas(V_SIZE) i64     i64[V_SIZE/sizeof(i64)];
        alignas(V_SIZE) u8      u8[V_SIZE/sizeof(u8)];
        alignas(V_SIZE) u16     u16[V_SIZE/sizeof(u16)];
        alignas(V_SIZE) u32     u32[V_SIZE/sizeof(u32)];
        alignas(V_SIZE) u64     u64[V_SIZE/sizeof(u64)];
        alignas(V_SIZE) f32     f32[V_SIZE/sizeof(f32)];
        alignas(V_SIZE) f64     f64[V_SIZE/sizeof(f64)];
        alignas(V_SIZE) TV vector;
    };
private:
    __m512_dbg(TV v) {
        vector = v;
    }
public:
    static __m512_dbg make(__m512i v);
};


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
