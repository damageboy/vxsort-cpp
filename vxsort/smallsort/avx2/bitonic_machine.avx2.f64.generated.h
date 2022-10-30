/////////////////////////////////////////////////////////////////////////////
////
// This file was auto-generated by a tool at 2022-10-30 08:12:25
//
// It is recommended you DO NOT directly edit this file but instead edit
// the code-generator that generated this source file instead.
/////////////////////////////////////////////////////////////////////////////

#ifndef BITONIC_MACHINE_AVX2_F64_H
#define BITONIC_MACHINE_AVX2_F64_H

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

template<> struct bitonic_machine<f64, AVX2> {
    static const i32 N = 4;
    static constexpr f64 MAX = std::numeric_limits<f64>::max();
public:
    typedef __m256d TV;
    typedef __m256d TMASK;

    static INLINE void sort_01v_ascending(TV& d01) {
        TV min, max, s;

        s = _mm256_shuffle_pd(d01, d01, 0b0'1'0'1);
        min = _mm256_min_pd(s, d01);
        max = _mm256_max_pd(s, d01);
        d01 = _mm256_blend_pd(min, max, 0b00000110);

        s = _mm256_permute4x64_pd(d01, 0b01'00'11'10);
        min = _mm256_min_pd(s, d01);
        max = _mm256_max_pd(s, d01);
        d01 = _mm256_blend_pd(min, max, 0b00001100);

        s = _mm256_shuffle_pd(d01, d01, 0b0'1'0'1);
        min = _mm256_min_pd(s, d01);
        max = _mm256_max_pd(s, d01);
        d01 = _mm256_blend_pd(min, max, 0b00001010);
    }
    static INLINE void merge_01v_ascending(TV& d01) {
        TV min, max, s;

        s = _mm256_permute4x64_pd(d01, 0b01'00'11'10);
        min = _mm256_min_pd(s, d01);
        max = _mm256_max_pd(s, d01);
        d01 = _mm256_blend_pd(min, max, 0b00001100);

        s = _mm256_shuffle_pd(d01, d01, 0b0'1'0'1);
        min = _mm256_min_pd(s, d01);
        max = _mm256_max_pd(s, d01);
        d01 = _mm256_blend_pd(min, max, 0b00001010);
    }
    static INLINE void sort_01v_descending(TV& d01) {
        TV min, max, s;

        s = _mm256_shuffle_pd(d01, d01, 0b0'1'0'1);
        min = _mm256_min_pd(s, d01);
        max = _mm256_max_pd(s, d01);
        d01 = _mm256_blend_pd(min, max, 0b00001001);

        s = _mm256_permute4x64_pd(d01, 0b01'00'11'10);
        min = _mm256_min_pd(s, d01);
        max = _mm256_max_pd(s, d01);
        d01 = _mm256_blend_pd(min, max, 0b00000011);

        s = _mm256_shuffle_pd(d01, d01, 0b0'1'0'1);
        min = _mm256_min_pd(s, d01);
        max = _mm256_max_pd(s, d01);
        d01 = _mm256_blend_pd(min, max, 0b00000101);
    }
    static INLINE void merge_01v_descending(TV& d01) {
        TV min, max, s;

        s = _mm256_permute4x64_pd(d01, 0b01'00'11'10);
        min = _mm256_min_pd(s, d01);
        max = _mm256_max_pd(s, d01);
        d01 = _mm256_blend_pd(min, max, 0b00000011);

        s = _mm256_shuffle_pd(d01, d01, 0b0'1'0'1);
        min = _mm256_min_pd(s, d01);
        max = _mm256_max_pd(s, d01);
        d01 = _mm256_blend_pd(min, max, 0b00000101);
    }
    static INLINE void sort_02v_ascending(TV& d01, TV& d02) {
        TV tmp;

        sort_01v_ascending(d01);
        sort_01v_descending(d02);

        tmp = d02;
        d02 = _mm256_max_pd(d01, d02);
        d01 = _mm256_min_pd(d01, tmp);

        merge_01v_ascending(d01);
        merge_01v_ascending(d02);
    }
    static INLINE void sort_02v_descending(TV& d01, TV& d02) {
        TV tmp;

        sort_01v_descending(d01);
        sort_01v_ascending(d02);

        tmp = d02;
        d02 = _mm256_max_pd(d01, d02);
        d01 = _mm256_min_pd(d01, tmp);

        merge_01v_descending(d01);
        merge_01v_descending(d02);
    }
    static INLINE void merge_02v_ascending(TV& d01, TV& d02) {
        TV tmp;

        tmp = d01;
        d01 = _mm256_min_pd(d02, d01);
        d02 = _mm256_max_pd(d02, tmp);

        merge_01v_ascending(d01);
        merge_01v_ascending(d02);
    }
    static INLINE void merge_02v_descending(TV& d01, TV& d02) {
        TV tmp;

        tmp = d01;
        d01 = _mm256_min_pd(d02, d01);
        d02 = _mm256_max_pd(d02, tmp);

        merge_01v_descending(d01);
        merge_01v_descending(d02);
    }
    static INLINE void sort_03v_ascending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        sort_02v_ascending(d01, d02);
        sort_01v_descending(d03);

        tmp = d03;
        d03 = _mm256_max_pd(d02, d03);
        d02 = _mm256_min_pd(d02, tmp);

        merge_02v_ascending(d01, d02);
        merge_01v_ascending(d03);
    }
    static INLINE void sort_03v_descending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        sort_02v_descending(d01, d02);
        sort_01v_ascending(d03);

        tmp = d03;
        d03 = _mm256_max_pd(d02, d03);
        d02 = _mm256_min_pd(d02, tmp);

        merge_02v_descending(d01, d02);
        merge_01v_descending(d03);
    }
    static INLINE void merge_03v_ascending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        tmp = d01;
        d01 = _mm256_min_pd(d03, d01);
        d03 = _mm256_max_pd(d03, tmp);

        merge_02v_ascending(d01, d02);
        merge_01v_ascending(d03);
    }
    static INLINE void merge_03v_descending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        tmp = d01;
        d01 = _mm256_min_pd(d03, d01);
        d03 = _mm256_max_pd(d03, tmp);

        merge_02v_descending(d01, d02);
        merge_01v_descending(d03);
    }
    static INLINE void sort_04v_ascending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        sort_02v_ascending(d01, d02);
        sort_02v_descending(d03, d04);

        tmp = d03;
        d03 = _mm256_max_pd(d02, d03);
        d02 = _mm256_min_pd(d02, tmp);

        tmp = d04;
        d04 = _mm256_max_pd(d01, d04);
        d01 = _mm256_min_pd(d01, tmp);

        merge_02v_ascending(d01, d02);
        merge_02v_ascending(d03, d04);
    }
    static INLINE void sort_04v_descending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        sort_02v_descending(d01, d02);
        sort_02v_ascending(d03, d04);

        tmp = d03;
        d03 = _mm256_max_pd(d02, d03);
        d02 = _mm256_min_pd(d02, tmp);

        tmp = d04;
        d04 = _mm256_max_pd(d01, d04);
        d01 = _mm256_min_pd(d01, tmp);

        merge_02v_descending(d01, d02);
        merge_02v_descending(d03, d04);
    }
    static INLINE void merge_04v_ascending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        tmp = d01;
        d01 = _mm256_min_pd(d03, d01);
        d03 = _mm256_max_pd(d03, tmp);

        tmp = d02;
        d02 = _mm256_min_pd(d04, d02);
        d04 = _mm256_max_pd(d04, tmp);

        merge_02v_ascending(d01, d02);
        merge_02v_ascending(d03, d04);
    }
    static INLINE void merge_04v_descending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        tmp = d01;
        d01 = _mm256_min_pd(d03, d01);
        d03 = _mm256_max_pd(d03, tmp);

        tmp = d02;
        d02 = _mm256_min_pd(d04, d02);
        d04 = _mm256_max_pd(d04, tmp);

        merge_02v_descending(d01, d02);
        merge_02v_descending(d03, d04);
    }
    static INLINE void cross_min_max(TV& d01, TV& d02) {
        TV tmp;

        tmp = _mm256_permute4x64_pd(d02, 0b00'01'10'11);
        d02 = _mm256_max_pd(d01, tmp);
        d01 = _mm256_min_pd(d01, tmp);
    }
    static INLINE void strided_min_max(TV& dl, TV& dr) {
        TV tmp;

        tmp = dl;
        dl = _mm256_min_pd(dr, dl);
        dr = _mm256_max_pd(dr, tmp);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_01v_full_ascending(f64 *ptr) {
        TV d01 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 0));
        sort_01v_ascending(d01);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 0), d01);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_02v_full_ascending(f64 *ptr) {
        TV d01 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 0));
        TV d02 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 1));
        sort_02v_ascending(d01, d02);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 0), d01);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 1), d02);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_03v_full_ascending(f64 *ptr) {
        TV d01 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 0));
        TV d02 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 1));
        TV d03 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 2));
        sort_03v_ascending(d01, d02, d03);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 0), d01);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 1), d02);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 2), d03);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_04v_full_ascending(f64 *ptr) {
        TV d01 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 0));
        TV d02 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 1));
        TV d03 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 2));
        TV d04 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 3));
        sort_04v_ascending(d01, d02, d03, d04);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 0), d01);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 1), d02);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 2), d03);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 3), d04);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_01v_full_descending(f64 *ptr) {
        TV d01 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 0));
        sort_01v_descending(d01);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 0), d01);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_02v_full_descending(f64 *ptr) {
        TV d01 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 0));
        TV d02 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 1));
        sort_02v_descending(d01, d02);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 0), d01);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 1), d02);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_03v_full_descending(f64 *ptr) {
        TV d01 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 0));
        TV d02 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 1));
        TV d03 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 2));
        sort_03v_descending(d01, d02, d03);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 0), d01);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 1), d02);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 2), d03);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_04v_full_descending(f64 *ptr) {
        TV d01 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 0));
        TV d02 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 1));
        TV d03 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 2));
        TV d04 = _mm256_loadu_pd((f64 const *) ((__m256d const *) ptr + 3));
        sort_04v_descending(d01, d02, d03, d04);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 0), d01);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 1), d02);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 2), d03);
        _mm256_storeu_pd((f64 *) ((__m256d *)  ptr + 3), d04);
    }

    // This is generated for testing purposes only
    static void sort_full_vectors_ascending(f64 *ptr, usize length) {
        assert(length % N == 0);
        switch(length / N) {
            case 1: sort_01v_full_ascending(ptr); break;
            case 2: sort_02v_full_ascending(ptr); break;
            case 3: sort_03v_full_ascending(ptr); break;
            case 4: sort_04v_full_ascending(ptr); break;
        }
    }

    // This is generated for testing purposes only
    static void sort_full_vectors_descending(f64 *ptr, usize length) {
        assert(length % N == 0);
        switch(length / N) {
            case 1: sort_01v_full_descending(ptr); break;
            case 2: sort_02v_full_descending(ptr); break;
            case 3: sort_03v_full_descending(ptr); break;
            case 4: sort_04v_full_descending(ptr); break;
        }
    }

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
