/////////////////////////////////////////////////////////////////////////////
////
// This file was auto-generated by a tool at 2022-10-08 19:28:11
//
// It is recommended you DO NOT directly edit this file but instead edit
// the code-generator that generated this source file instead.
/////////////////////////////////////////////////////////////////////////////

#ifndef BITONIC_MACHINE_AVX2_U64_H
#define BITONIC_MACHINE_AVX2_U64_H

#include "../../vxsort_targets_enable_avx2.h"

#include <cassert>
#include <limits>
#include <immintrin.h>
#include "../bitonic_machine.h"

#define i2d _mm256_castsi256_pd
#define d2i _mm256_castpd_si256
#define i2s _mm256_castsi256_ps
#define s2i _mm256_castps_si256
#define s2d _mm256_castps_pd
#define d2s _mm256_castpd_ps

namespace vxsort {
namespace smallsort {
using namespace vxsort::types;

template<> struct bitonic_machine<u64, AVX2> {
    static const int N = 4;
    static constexpr u64 MAX = std::numeric_limits<u64>::max();
public:
    typedef __m256i TV;
    typedef __m256i TMASK;

    static INLINE void sort_01v_ascending(TV& d01) {
        TV min, max, s;
        TV cmp;
        const TV topBit = _mm256_set1_epi64x(1LLU << 63);

        s = d2i(_mm256_shuffle_pd(i2d(d01), i2d(d01), 0b0'1'0'1));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00000110));

        s = d2i(_mm256_permute4x64_pd(i2d(d01), 0b01'00'11'10));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00001100));

        s = d2i(_mm256_shuffle_pd(i2d(d01), i2d(d01), 0b0'1'0'1));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00001010));
    }
    static INLINE void merge_01v_ascending(TV& d01) {
        TV min, max, s;
        TV cmp;
        const TV topBit = _mm256_set1_epi64x(1LLU << 63);

        s = d2i(_mm256_permute4x64_pd(i2d(d01), 0b01'00'11'10));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00001100));

        s = d2i(_mm256_shuffle_pd(i2d(d01), i2d(d01), 0b0'1'0'1));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00001010));
    }
    static INLINE void sort_01v_descending(TV& d01) {
        TV min, max, s;
        TV cmp;
        const TV topBit = _mm256_set1_epi64x(1LLU << 63);

        s = d2i(_mm256_shuffle_pd(i2d(d01), i2d(d01), 0b0'1'0'1));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00001001));

        s = d2i(_mm256_permute4x64_pd(i2d(d01), 0b01'00'11'10));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00000011));

        s = d2i(_mm256_shuffle_pd(i2d(d01), i2d(d01), 0b0'1'0'1));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00000101));
    }
    static INLINE void merge_01v_descending(TV& d01) {
        TV min, max, s;
        TV cmp;
        const TV topBit = _mm256_set1_epi64x(1LLU << 63);

        s = d2i(_mm256_permute4x64_pd(i2d(d01), 0b01'00'11'10));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00000011));

        s = d2i(_mm256_shuffle_pd(i2d(d01), i2d(d01), 0b0'1'0'1));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00000101));
    }
    static INLINE void sort_02v_ascending(TV& d01, TV& d02) {
        TV tmp, cmp;
        TV topBit = _mm256_set1_epi64x(1LLU << 63);

        sort_01v_ascending(d01);
        sort_01v_descending(d02);

        tmp = d02;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d01), _mm256_xor_si256(topBit, d02));
        d02 = d2i(_mm256_blendv_pd(i2d(d02), i2d(d01), i2d(cmp)));
        d01 = d2i(_mm256_blendv_pd(i2d(d01), i2d(tmp), i2d(cmp)));

        merge_01v_ascending(d01);
        merge_01v_ascending(d02);
    }
    static INLINE void sort_02v_descending(TV& d01, TV& d02) {
        TV tmp, cmp;
        TV topBit = _mm256_set1_epi64x(1LLU << 63);

        sort_01v_descending(d01);
        sort_01v_ascending(d02);

        tmp = d02;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d01), _mm256_xor_si256(topBit, d02));
        d02 = d2i(_mm256_blendv_pd(i2d(d02), i2d(d01), i2d(cmp)));
        d01 = d2i(_mm256_blendv_pd(i2d(d01), i2d(tmp), i2d(cmp)));

        merge_01v_descending(d01);
        merge_01v_descending(d02);
    }
    static INLINE void merge_02v_ascending(TV& d01, TV& d02) {
        TV tmp, cmp;
        TV topBit = _mm256_set1_epi64x(1LLU << 63);

        tmp = d01;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d01));
        d01 = d2i(_mm256_blendv_pd(i2d(d02), i2d(d01), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, tmp));
        d02 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d02), i2d(cmp)));

        merge_01v_ascending(d01);
        merge_01v_ascending(d02);
    }
    static INLINE void merge_02v_descending(TV& d01, TV& d02) {
        TV tmp, cmp;
        TV topBit = _mm256_set1_epi64x(1LLU << 63);

        tmp = d01;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d01));
        d01 = d2i(_mm256_blendv_pd(i2d(d02), i2d(d01), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, tmp));
        d02 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d02), i2d(cmp)));

        merge_01v_descending(d01);
        merge_01v_descending(d02);
    }
    static INLINE void sort_03v_ascending(TV& d01, TV& d02, TV& d03) {
        TV tmp, cmp;
        TV topBit = _mm256_set1_epi64x(1LLU << 63);

        sort_02v_ascending(d01, d02);
        sort_01v_descending(d03);

        tmp = d03;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d03));
        d03 = d2i(_mm256_blendv_pd(i2d(d03), i2d(d02), i2d(cmp)));
        d02 = d2i(_mm256_blendv_pd(i2d(d02), i2d(tmp), i2d(cmp)));

        merge_02v_ascending(d01, d02);
        merge_01v_ascending(d03);
    }
    static INLINE void sort_03v_descending(TV& d01, TV& d02, TV& d03) {
        TV tmp, cmp;
        TV topBit = _mm256_set1_epi64x(1LLU << 63);

        sort_02v_descending(d01, d02);
        sort_01v_ascending(d03);

        tmp = d03;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d03));
        d03 = d2i(_mm256_blendv_pd(i2d(d03), i2d(d02), i2d(cmp)));
        d02 = d2i(_mm256_blendv_pd(i2d(d02), i2d(tmp), i2d(cmp)));

        merge_02v_descending(d01, d02);
        merge_01v_descending(d03);
    }
    static INLINE void merge_03v_ascending(TV& d01, TV& d02, TV& d03) {
        TV tmp, cmp;
        TV topBit = _mm256_set1_epi64x(1LLU << 63);

        tmp = d01;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d01));
        d01 = d2i(_mm256_blendv_pd(i2d(d03), i2d(d01), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, tmp));
        d03 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d03), i2d(cmp)));

        merge_02v_ascending(d01, d02);
        merge_01v_ascending(d03);
    }
    static INLINE void merge_03v_descending(TV& d01, TV& d02, TV& d03) {
        TV tmp, cmp;
        TV topBit = _mm256_set1_epi64x(1LLU << 63);

        tmp = d01;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d01));
        d01 = d2i(_mm256_blendv_pd(i2d(d03), i2d(d01), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, tmp));
        d03 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d03), i2d(cmp)));

        merge_02v_descending(d01, d02);
        merge_01v_descending(d03);
    }
    static INLINE void sort_04v_ascending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp, cmp;
        TV topBit = _mm256_set1_epi64x(1LLU << 63);

        sort_02v_ascending(d01, d02);
        sort_02v_descending(d03, d04);

        tmp = d03;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d03));
        d03 = d2i(_mm256_blendv_pd(i2d(d03), i2d(d02), i2d(cmp)));
        d02 = d2i(_mm256_blendv_pd(i2d(d02), i2d(tmp), i2d(cmp)));

        tmp = d04;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d01), _mm256_xor_si256(topBit, d04));
        d04 = d2i(_mm256_blendv_pd(i2d(d04), i2d(d01), i2d(cmp)));
        d01 = d2i(_mm256_blendv_pd(i2d(d01), i2d(tmp), i2d(cmp)));

        merge_02v_ascending(d01, d02);
        merge_02v_ascending(d03, d04);
    }
    static INLINE void sort_04v_descending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp, cmp;
        TV topBit = _mm256_set1_epi64x(1LLU << 63);

        sort_02v_descending(d01, d02);
        sort_02v_ascending(d03, d04);

        tmp = d03;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d03));
        d03 = d2i(_mm256_blendv_pd(i2d(d03), i2d(d02), i2d(cmp)));
        d02 = d2i(_mm256_blendv_pd(i2d(d02), i2d(tmp), i2d(cmp)));

        tmp = d04;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d01), _mm256_xor_si256(topBit, d04));
        d04 = d2i(_mm256_blendv_pd(i2d(d04), i2d(d01), i2d(cmp)));
        d01 = d2i(_mm256_blendv_pd(i2d(d01), i2d(tmp), i2d(cmp)));

        merge_02v_descending(d01, d02);
        merge_02v_descending(d03, d04);
    }
    static INLINE void merge_04v_ascending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp, cmp;
        TV topBit = _mm256_set1_epi64x(1LLU << 63);

        tmp = d01;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d01));
        d01 = d2i(_mm256_blendv_pd(i2d(d03), i2d(d01), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, tmp));
        d03 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d03), i2d(cmp)));

        tmp = d02;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d02));
        d02 = d2i(_mm256_blendv_pd(i2d(d04), i2d(d02), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, tmp));
        d04 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d04), i2d(cmp)));

        merge_02v_ascending(d01, d02);
        merge_02v_ascending(d03, d04);
    }
    static INLINE void merge_04v_descending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp, cmp;
        TV topBit = _mm256_set1_epi64x(1LLU << 63);

        tmp = d01;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d01));
        d01 = d2i(_mm256_blendv_pd(i2d(d03), i2d(d01), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, tmp));
        d03 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d03), i2d(cmp)));

        tmp = d02;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d02));
        d02 = d2i(_mm256_blendv_pd(i2d(d04), i2d(d02), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, tmp));
        d04 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d04), i2d(cmp)));

        merge_02v_descending(d01, d02);
        merge_02v_descending(d03, d04);
    }
    static INLINE void cross_min_max(TV& d01, TV& d02) {
        TV tmp;
        TV cmp;
        const TV topBit = _mm256_set1_epi64x(1LLU << 63);

        tmp = d2i(_mm256_permute4x64_pd(i2d(d02), 0b00'01'10'11));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d01), _mm256_xor_si256(topBit, tmp));
        d02 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d01), i2d(cmp)));
        d01 = d2i(_mm256_blendv_pd(i2d(d01), i2d(tmp), i2d(cmp)));
    }
    static INLINE void strided_min_max(TV& dl, TV& dr) {
        TV tmp;
        TV cmp;
        const TV topBit = _mm256_set1_epi64x(1LLU << 63);

        tmp = dl;
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, dr), _mm256_xor_si256(topBit, dl));
        dl = d2i(_mm256_blendv_pd(i2d(dr), i2d(dl), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, dr), _mm256_xor_si256(topBit, tmp));
        dr = d2i(_mm256_blendv_pd(i2d(tmp), i2d(dr), i2d(cmp)));
    }

#ifdef BITONIC_TESTS

    // This is generated for testing purposes only
    static NOINLINE void sort_01v_full_ascending(u64 *ptr) {
        TV d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
        sort_01v_ascending(d01);
        _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_02v_full_ascending(u64 *ptr) {
        TV d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
        TV d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);;
        sort_02v_ascending(d01, d02);
        _mm256_storeu_si256((__m256i *) ptr + 0, d01);
        _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_03v_full_ascending(u64 *ptr) {
        TV d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
        TV d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);;
        TV d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);;
        sort_03v_ascending(d01, d02, d03);
        _mm256_storeu_si256((__m256i *) ptr + 0, d01);
        _mm256_storeu_si256((__m256i *) ptr + 1, d02);
        _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_04v_full_ascending(u64 *ptr) {
        TV d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
        TV d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);;
        TV d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);;
        TV d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);;
        sort_04v_ascending(d01, d02, d03, d04);
        _mm256_storeu_si256((__m256i *) ptr + 0, d01);
        _mm256_storeu_si256((__m256i *) ptr + 1, d02);
        _mm256_storeu_si256((__m256i *) ptr + 2, d03);
        _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_01v_full_descending(u64 *ptr) {
        TV d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
        sort_01v_descending(d01);
        _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_02v_full_descending(u64 *ptr) {
        TV d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
        TV d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);;
        sort_02v_descending(d01, d02);
        _mm256_storeu_si256((__m256i *) ptr + 0, d01);
        _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_03v_full_descending(u64 *ptr) {
        TV d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
        TV d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);;
        TV d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);;
        sort_03v_descending(d01, d02, d03);
        _mm256_storeu_si256((__m256i *) ptr + 0, d01);
        _mm256_storeu_si256((__m256i *) ptr + 1, d02);
        _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_04v_full_descending(u64 *ptr) {
        TV d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
        TV d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);;
        TV d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);;
        TV d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);;
        sort_04v_descending(d01, d02, d03, d04);
        _mm256_storeu_si256((__m256i *) ptr + 0, d01);
        _mm256_storeu_si256((__m256i *) ptr + 1, d02);
        _mm256_storeu_si256((__m256i *) ptr + 2, d03);
        _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    }

    // This is generated for testing purposes only
    static void sort_full_vectors_ascending(u64 *ptr, size_t length) {
        assert(length % N == 0);
        switch(length / N) {
            case 1: sort_01v_full_ascending(ptr); break;
            case 2: sort_02v_full_ascending(ptr); break;
            case 3: sort_03v_full_ascending(ptr); break;
            case 4: sort_04v_full_ascending(ptr); break;
        }
    }

    // This is generated for testing purposes only
    static void sort_full_vectors_descending(u64 *ptr, size_t length) {
        assert(length % N == 0);
        switch(length / N) {
            case 1: sort_01v_full_descending(ptr); break;
            case 2: sort_02v_full_descending(ptr); break;
            case 3: sort_03v_full_descending(ptr); break;
            case 4: sort_04v_full_descending(ptr); break;
        }
    }

#endif

};
}
}

#undef i2d
#undef d2i
#undef i2s
#undef s2i
#undef s2d
#undef d2s

#include "../../vxsort_targets_disable.h"

#endif
