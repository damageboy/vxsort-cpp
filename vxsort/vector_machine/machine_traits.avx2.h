#ifndef VXSORT_MACHINE_TRAITS_AVX2_H
#define VXSORT_MACHINE_TRAITS_AVX2_H

#include "vxsort_targets_enable_avx2.h"

#include <stdexcept>
#include <array>
#include <algorithm>
#include <type_traits>
#include <limits>
#include <cassert>
#include <cstring>
#include <cinttypes>
#include <immintrin.h>

#include "defs.h"
#include "machine_traits.h"

#define i2d _mm256_castsi256_pd
#define d2i _mm256_castpd_si256
#define i2s _mm256_castsi256_ps
#define s2i _mm256_castps_si256
#define s2d _mm256_castps_pd
#define d2s _mm256_castpd_ps

namespace vxsort {
using namespace vxsort::types;

// * We might read the last 4 bytes into a 128-bit vector for 64-bit element masking
// * We might read the last 8 bytes into a 128-bit vector for 32-bit element masking
// This mostly applies to debug mode, since without optimizations, most compilers
// actually execute the instruction stream _mm256_cvtepi8_epiNN + _mm_loadu_si128 as they are given.
// In contrast, release/optimizing compilers, turn that very specific intrinsic pair to
// a more reasonable: vpmovsxbq ymm0, dword [rax*4 + mask_table_4], eliminating the 128-bit
// load completely and effectively reading exactly 4/8 (depending if the instruction is vpmovsxb[q,d]
// without generating an out of bounds read at all.
// But, life is harsh, and we can't trust the compiler to do the right thing if it is not
// contractual, hence this flustercuck
const int M4_SIZE = 16 + 12;
const int M8_SIZE = 64 + 8;

extern const u8 mask_table_4[M4_SIZE];
extern const u8 mask_table_8[M8_SIZE];


extern const i8 perm_table_64[128];
extern const i8 perm_table_32[2048];

struct __m256_dbg {
    using TV = __m256i;
    static const int V_SIZE = sizeof(__m256i);
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
    __m256_dbg(TV v) {
        vector = v;
    }
   public:
    static __m256_dbg make(__m256i v);
};

#include "avx2/f64.h"
#include "avx2/f32.h"
#include "avx2/i16.h"
#include "avx2/i32.h"
#include "avx2/i64.h"
#include "avx2/u16.h"
#include "avx2/u32.h"
#include "avx2/u64.h"
}

#undef i2d
#undef d2i
#undef i2s
#undef s2i
#undef s2d
#undef d2s

#include "vxsort_targets_disable.h"
#endif  // VXSORT_VXSORT_AVX2_H
