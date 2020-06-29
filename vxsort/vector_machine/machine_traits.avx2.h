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
