//
// Created by dans on 6/1/20.
//

#ifndef VXSORT_MACHINE_TRAITS_AVX512_H
#define VXSORT_MACHINE_TRAITS_AVX512_H

#include "vxsort_targets_enable_avx512.h"

#include <immintrin.h>
#include "defs.h"
#include "machine_traits.h"

namespace vxsort {
template <>
class vxsort_machine_traits<int32_t, AVX512> {
   public:
    typedef __m512i TV;
    typedef __mmask16 TMASK;

    static constexpr bool supports_compress_writes() { return true; }

    static INLINE TV load_vec(TV* p) { return _mm512_loadu_si512(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm512_storeu_si512(ptr, v); }

    // Will never be called
    static INLINE TV partition_vector(TV v, int mask) { return v; }

    static void store_compress_vec(TV* ptr, TV v, TMASK mask) { _mm512_mask_compressstoreu_epi32(ptr, mask, v); }

    static INLINE TV broadcast(int32_t pivot) { return _mm512_set1_epi32(pivot); }

    static INLINE TMASK get_cmpgt_mask(TV a, TV b) { return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_GT); }

    static TV shift_right(TV v, int i) { return _mm512_srli_epi32(v, i); }
    static TV shift_left(TV v, int i) { return _mm512_slli_epi32(v, i); }

    static INLINE TV add(TV a, TV b) { return _mm512_add_epi32(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm512_sub_epi32(a, b); };
};

template <>
class vxsort_machine_traits<uint32_t, AVX512> {
   public:
    typedef __m512i TV;
    typedef __mmask16 TMASK;

    static constexpr bool supports_compress_writes() { return true; }

    static INLINE TV load_vec(TV* p) { return _mm512_loadu_si512(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm512_storeu_si512(ptr, v); }

    // Will never be called
    static INLINE TV partition_vector(TV v, int mask) { return v; }

    static void store_compress_vec(TV* ptr, TV v, TMASK mask) { _mm512_mask_compressstoreu_epi32(ptr, mask, v); }

    static INLINE TV broadcast(uint32_t pivot) { return _mm512_set1_epi32(pivot); }

    static INLINE TMASK get_cmpgt_mask(TV a, TV b) { return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_GT); }

    static TV shift_right(TV v, int i) { return _mm512_srli_epi32(v, i); }
    static TV shift_left(TV v, int i) { return _mm512_slli_epi32(v, i); }

    static INLINE TV add(TV a, TV b) { return _mm512_add_epi32(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm512_sub_epi32(a, b); };
};

template <>
class vxsort_machine_traits<float, AVX512> {
   public:
    typedef __m512 TV;
    typedef __mmask16 TMASK;

    static constexpr bool supports_compress_writes() { return true; }

    static INLINE TV load_vec(TV* p) { return _mm512_loadu_ps(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm512_storeu_ps(ptr, v); }

    // Will never be called
    static INLINE TV partition_vector(TV v, int mask) { return v; }

    static void store_compress_vec(TV* ptr, TV v, TMASK mask) { _mm512_mask_compressstoreu_ps(ptr, mask, v); }

    static INLINE TV broadcast(float pivot) { return _mm512_set1_ps(pivot); }

    static INLINE TMASK get_cmpgt_mask(TV a, TV b) { return _mm512_cmp_ps_mask(a, b, _CMP_GT_OS); }
};

template <>
class vxsort_machine_traits<int64_t, AVX512> {
   public:
    typedef __m512i TV;
    typedef __mmask8 TMASK;

    static constexpr bool supports_compress_writes() { return true; }

    static INLINE TV load_vec(TV* p) { return _mm512_loadu_si512(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm512_storeu_si512(ptr, v); }

    // Will never be called
    static INLINE TV partition_vector(TV v, int mask) { return v; }

    static void store_compress_vec(TV* ptr, TV v, TMASK mask) { _mm512_mask_compressstoreu_epi64(ptr, mask, v); }

    static INLINE TV broadcast(int64_t pivot) { return _mm512_set1_epi64(pivot); }

    static INLINE TMASK get_cmpgt_mask(TV a, TV b) { return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_GT); }

    static TV shift_right(TV v, int i) { return _mm512_srli_epi64(v, i); }
    static TV shift_left(TV v, int i) { return _mm512_slli_epi64(v, i); }

    static INLINE TV add(TV a, TV b) { return _mm512_add_epi64(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm512_sub_epi64(a, b); };

    static INLINE TV pack_ordered(TV a, TV b) {
        a = _mm512_permutex_epi64(_mm512_shuffle_epi32(a, _MM_PERM_DBCA), _MM_PERM_DBCA);
        b = _mm512_permutex_epi64(_mm512_shuffle_epi32(b, _MM_PERM_DBCA), _MM_PERM_CADB);
        return _mm512_shuffle_i64x2(a, b, _MM_PERM_DBCA);
    }

    static INLINE TV pack_unordered(TV a, TV b) { return _mm512_mask_shuffle_epi32(a, 0b1010101010101010, b, _MM_PERM_CDAB); }

    static INLINE void unpack_ordered(TV p, TV& u1, TV& u2) {
        auto p01 = _mm512_extracti32x8_epi32(p, 0);
        auto p02 = _mm512_extracti32x8_epi32(p, 1);

        u1 = _mm512_cvtepi32_epi64(p01);
        u2 = _mm512_cvtepi32_epi64(p02);
    }
};

template <>
class vxsort_machine_traits<uint64_t, AVX512> {
   public:
    typedef __m512i TV;
    typedef __mmask8 TMASK;

    static constexpr bool supports_compress_writes() { return true; }

    static INLINE TV load_vec(TV* p) { return _mm512_loadu_si512(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm512_storeu_si512(ptr, v); }

    // Will never be called
    static INLINE TV partition_vector(TV v, int mask) { return v; }

    static void store_compress_vec(TV* ptr, TV v, TMASK mask) { _mm512_mask_compressstoreu_epi64(ptr, mask, v); }

    static INLINE TV broadcast(uint64_t pivot) { return _mm512_set1_epi64(pivot); }

    static INLINE TMASK get_cmpgt_mask(TV a, TV b) { return _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_GT); }

    static TV shift_right(TV v, int i) { return _mm512_srli_epi64(v, i); }
    static TV shift_left(TV v, int i) { return _mm512_slli_epi64(v, i); }

    static INLINE TV add(TV a, TV b) { return _mm512_add_epi64(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm512_sub_epi64(a, b); };
};

template <>
class vxsort_machine_traits<double, AVX512> {
   public:
    typedef __m512d TV;
    typedef __mmask8 TMASK;

    static constexpr bool supports_compress_writes() { return true; }

    static INLINE TV load_vec(TV* p) { return _mm512_loadu_pd(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm512_storeu_pd(ptr, v); }

    // Will never be called
    static INLINE TV partition_vector(TV v, int mask) { return v; }

    static void store_compress_vec(TV* ptr, TV v, TMASK mask) { _mm512_mask_compressstoreu_pd(ptr, mask, v); }

    static INLINE TV broadcast(double pivot) { return _mm512_set1_pd(pivot); }

    static INLINE TMASK get_cmpgt_mask(TV a, TV b) { return _mm512_cmp_pd_mask(a, b, _CMP_GT_OS); }
};

}

#include "vxsort_targets_disable.h"

#endif  // VXSORT_VXSORT_AVX512_H
