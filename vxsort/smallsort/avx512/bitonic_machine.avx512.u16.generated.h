/////////////////////////////////////////////////////////////////////////////
////
// This file was auto-generated by a tool at 2022-10-30 08:12:25
//
// It is recommended you DO NOT directly edit this file but instead edit
// the code-generator that generated this source file instead.
/////////////////////////////////////////////////////////////////////////////

#ifndef BITONIC_MACHINE_AVX512_U16_H
#define BITONIC_MACHINE_AVX512_U16_H

#include "../../vxsort_targets_enable_avx512.h"

#include <limits>
#include <immintrin.h>
#include "../bitonic_machine.h"

#define i2d _mm512_castsi512_pd
#define d2i _mm512_castpd_si512
#define i2s _mm512_castsi512_ps
#define s2i _mm512_castps_si512
#define s2d _mm512_castps_pd
#define d2s _mm521_castpd_ps

namespace vxsort {
namespace smallsort {
using namespace vxsort::types;

template<> struct bitonic_machine<u16, AVX512> {
    static const i32 N = 32;
    static constexpr u16 MAX = std::numeric_limits<u16>::max();
public:
    typedef __m512i TV;
    typedef __mmask32 TMASK;

    static INLINE void sort_01v_ascending(TV& d01) {
        TV  min, s;
        const TV x1 = _mm512_set_epi64(0x1D1C1F1E19181B1A, 0x1514171611101312, 0xD0C0F0E09080B0A, 0x504070601000302,
                               0x3D3C3F3E39383B3A, 0x3534373631303332, 0x2D2C2F2E29282B2A, 0x2524272621202322);

        s = _mm512_shuffle_epi8(d01, x1);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b01100110011001100110011001100110, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b00111100001111000011110000111100, s, d01);

        s = _mm512_shuffle_epi8(d01, x1);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b01011010010110100101101001011010, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b00001111111100000000111111110000, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b00110011110011000011001111001100, s, d01);

        s = _mm512_shuffle_epi8(d01, x1);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b01010101101010100101010110101010, s, d01);

        s = _mm512_permutex_epi64(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b00000000111111111111111100000000, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b00001111000011111111000011110000, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b00110011001100111100110011001100, s, d01);

        s = _mm512_shuffle_epi8(d01, x1);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b01010101010101011010101010101010, s, d01);

        s = _mm512_shuffle_i64x2(d01, d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b11111111111111110000000000000000, s, d01);

        s = _mm512_permutex_epi64(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b11111111000000001111111100000000, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b11110000111100001111000011110000, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b11001100110011001100110011001100, s, d01);

        s = _mm512_shuffle_epi8(d01, x1);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b10101010101010101010101010101010, s, d01);
    }
    static INLINE void merge_01v_ascending(TV& d01) {
        TV  min, s;
        const TV x1 = _mm512_set_epi64(0x1D1C1F1E19181B1A, 0x1514171611101312, 0xD0C0F0E09080B0A, 0x504070601000302,
                               0x3D3C3F3E39383B3A, 0x3534373631303332, 0x2D2C2F2E29282B2A, 0x2524272621202322);

        s = _mm512_shuffle_i64x2(d01, d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b11111111111111110000000000000000, s, d01);

        s = _mm512_permutex_epi64(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b11111111000000001111111100000000, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b11110000111100001111000011110000, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b11001100110011001100110011001100, s, d01);

        s = _mm512_shuffle_epi8(d01, x1);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b10101010101010101010101010101010, s, d01);
    }
    static INLINE void sort_01v_descending(TV& d01) {
        TV  min, s;
        const TV x1 = _mm512_set_epi64(0x1D1C1F1E19181B1A, 0x1514171611101312, 0xD0C0F0E09080B0A, 0x504070601000302,
                               0x3D3C3F3E39383B3A, 0x3534373631303332, 0x2D2C2F2E29282B2A, 0x2524272621202322);

        s = _mm512_shuffle_epi8(d01, x1);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b10011001100110011001100110011001, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b11000011110000111100001111000011, s, d01);

        s = _mm512_shuffle_epi8(d01, x1);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b10100101101001011010010110100101, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b11110000000011111111000000001111, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b11001100001100111100110000110011, s, d01);

        s = _mm512_shuffle_epi8(d01, x1);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b10101010010101011010101001010101, s, d01);

        s = _mm512_permutex_epi64(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b11111111000000000000000011111111, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b11110000111100000000111100001111, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b11001100110011000011001100110011, s, d01);

        s = _mm512_shuffle_epi8(d01, x1);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b10101010101010100101010101010101, s, d01);

        s = _mm512_shuffle_i64x2(d01, d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b00000000000000001111111111111111, s, d01);

        s = _mm512_permutex_epi64(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b00000000111111110000000011111111, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b00001111000011110000111100001111, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b00110011001100110011001100110011, s, d01);

        s = _mm512_shuffle_epi8(d01, x1);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b01010101010101010101010101010101, s, d01);
    }
    static INLINE void merge_01v_descending(TV& d01) {
        TV  min, s;
        const TV x1 = _mm512_set_epi64(0x1D1C1F1E19181B1A, 0x1514171611101312, 0xD0C0F0E09080B0A, 0x504070601000302,
                               0x3D3C3F3E39383B3A, 0x3534373631303332, 0x2D2C2F2E29282B2A, 0x2524272621202322);

        s = _mm512_shuffle_i64x2(d01, d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b00000000000000001111111111111111, s, d01);

        s = _mm512_permutex_epi64(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b00000000111111110000000011111111, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b00001111000011110000111100001111, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b00110011001100110011001100110011, s, d01);

        s = _mm512_shuffle_epi8(d01, x1);
        min = _mm512_min_epu16(s, d01);
        d01 = _mm512_mask_max_epu16(min, 0b01010101010101010101010101010101, s, d01);
    }
    static INLINE void sort_02v_ascending(TV& d01, TV& d02) {
        TV tmp;

        sort_01v_ascending(d01);
        sort_01v_descending(d02);

        tmp = d02;
        d02 = _mm512_max_epu16(d01, d02);
        d01 = _mm512_min_epu16(d01, tmp);

        merge_01v_ascending(d01);
        merge_01v_ascending(d02);
    }
    static INLINE void sort_02v_descending(TV& d01, TV& d02) {
        TV tmp;

        sort_01v_descending(d01);
        sort_01v_ascending(d02);

        tmp = d02;
        d02 = _mm512_max_epu16(d01, d02);
        d01 = _mm512_min_epu16(d01, tmp);

        merge_01v_descending(d01);
        merge_01v_descending(d02);
    }
    static INLINE void merge_02v_ascending(TV& d01, TV& d02) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epu16(d02, d01);
        d02 = _mm512_max_epu16(d02, tmp);

        merge_01v_ascending(d01);
        merge_01v_ascending(d02);
    }
    static INLINE void merge_02v_descending(TV& d01, TV& d02) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epu16(d02, d01);
        d02 = _mm512_max_epu16(d02, tmp);

        merge_01v_descending(d01);
        merge_01v_descending(d02);
    }
    static INLINE void sort_03v_ascending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        sort_02v_ascending(d01, d02);
        sort_01v_descending(d03);

        tmp = d03;
        d03 = _mm512_max_epu16(d02, d03);
        d02 = _mm512_min_epu16(d02, tmp);

        merge_02v_ascending(d01, d02);
        merge_01v_ascending(d03);
    }
    static INLINE void sort_03v_descending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        sort_02v_descending(d01, d02);
        sort_01v_ascending(d03);

        tmp = d03;
        d03 = _mm512_max_epu16(d02, d03);
        d02 = _mm512_min_epu16(d02, tmp);

        merge_02v_descending(d01, d02);
        merge_01v_descending(d03);
    }
    static INLINE void merge_03v_ascending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epu16(d03, d01);
        d03 = _mm512_max_epu16(d03, tmp);

        merge_02v_ascending(d01, d02);
        merge_01v_ascending(d03);
    }
    static INLINE void merge_03v_descending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epu16(d03, d01);
        d03 = _mm512_max_epu16(d03, tmp);

        merge_02v_descending(d01, d02);
        merge_01v_descending(d03);
    }
    static INLINE void sort_04v_ascending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        sort_02v_ascending(d01, d02);
        sort_02v_descending(d03, d04);

        tmp = d03;
        d03 = _mm512_max_epu16(d02, d03);
        d02 = _mm512_min_epu16(d02, tmp);

        tmp = d04;
        d04 = _mm512_max_epu16(d01, d04);
        d01 = _mm512_min_epu16(d01, tmp);

        merge_02v_ascending(d01, d02);
        merge_02v_ascending(d03, d04);
    }
    static INLINE void sort_04v_descending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        sort_02v_descending(d01, d02);
        sort_02v_ascending(d03, d04);

        tmp = d03;
        d03 = _mm512_max_epu16(d02, d03);
        d02 = _mm512_min_epu16(d02, tmp);

        tmp = d04;
        d04 = _mm512_max_epu16(d01, d04);
        d01 = _mm512_min_epu16(d01, tmp);

        merge_02v_descending(d01, d02);
        merge_02v_descending(d03, d04);
    }
    static INLINE void merge_04v_ascending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epu16(d03, d01);
        d03 = _mm512_max_epu16(d03, tmp);

        tmp = d02;
        d02 = _mm512_min_epu16(d04, d02);
        d04 = _mm512_max_epu16(d04, tmp);

        merge_02v_ascending(d01, d02);
        merge_02v_ascending(d03, d04);
    }
    static INLINE void merge_04v_descending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epu16(d03, d01);
        d03 = _mm512_max_epu16(d03, tmp);

        tmp = d02;
        d02 = _mm512_min_epu16(d04, d02);
        d04 = _mm512_max_epu16(d04, tmp);

        merge_02v_descending(d01, d02);
        merge_02v_descending(d03, d04);
    }
    static INLINE void cross_min_max(TV& d01, TV& d02) {
        TV tmp;
        const TV x1 = _mm512_set_epi64(0x1D1C1F1E19181B1A, 0x1514171611101312, 0xD0C0F0E09080B0A, 0x504070601000302,
                               0x3D3C3F3E39383B3A, 0x3534373631303332, 0x2D2C2F2E29282B2A, 0x2524272621202322);

        tmp = _mm512_shuffle_i64x2(_mm512_shuffle_epi32(_mm512_shuffle_epi8(d02, x1), (_MM_PERM_ENUM) 0b00'01'10'11), _mm512_shuffle_epi32(_mm512_shuffle_epi8(d02, x1), (_MM_PERM_ENUM) 0b00'01'10'11), (_MM_PERM_ENUM) 0b00'01'10'11);
        d02 = _mm512_max_epu16(d01, tmp);
        d01 = _mm512_min_epu16(d01, tmp);
    }
    static INLINE void strided_min_max(TV& d01, TV& d02) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epu16(d02, d01);
        d02 = _mm512_max_epu16(d02, tmp);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_01v_full_ascending(u16 *ptr) {
        TV d01 = _mm512_loadu_si512((__m512i const *) ptr + 0);;
        sort_01v_ascending(d01);
        _mm512_storeu_si512((__m512i *) ptr + 0, d01);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_02v_full_ascending(u16 *ptr) {
        TV d01 = _mm512_loadu_si512((__m512i const *) ptr + 0);;
        TV d02 = _mm512_loadu_si512((__m512i const *) ptr + 1);;
        sort_02v_ascending(d01, d02);
        _mm512_storeu_si512((__m512i *) ptr + 0, d01);
        _mm512_storeu_si512((__m512i *) ptr + 1, d02);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_03v_full_ascending(u16 *ptr) {
        TV d01 = _mm512_loadu_si512((__m512i const *) ptr + 0);;
        TV d02 = _mm512_loadu_si512((__m512i const *) ptr + 1);;
        TV d03 = _mm512_loadu_si512((__m512i const *) ptr + 2);;
        sort_03v_ascending(d01, d02, d03);
        _mm512_storeu_si512((__m512i *) ptr + 0, d01);
        _mm512_storeu_si512((__m512i *) ptr + 1, d02);
        _mm512_storeu_si512((__m512i *) ptr + 2, d03);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_04v_full_ascending(u16 *ptr) {
        TV d01 = _mm512_loadu_si512((__m512i const *) ptr + 0);;
        TV d02 = _mm512_loadu_si512((__m512i const *) ptr + 1);;
        TV d03 = _mm512_loadu_si512((__m512i const *) ptr + 2);;
        TV d04 = _mm512_loadu_si512((__m512i const *) ptr + 3);;
        sort_04v_ascending(d01, d02, d03, d04);
        _mm512_storeu_si512((__m512i *) ptr + 0, d01);
        _mm512_storeu_si512((__m512i *) ptr + 1, d02);
        _mm512_storeu_si512((__m512i *) ptr + 2, d03);
        _mm512_storeu_si512((__m512i *) ptr + 3, d04);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_01v_full_descending(u16 *ptr) {
        TV d01 = _mm512_loadu_si512((__m512i const *) ptr + 0);;
        sort_01v_descending(d01);
        _mm512_storeu_si512((__m512i *) ptr + 0, d01);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_02v_full_descending(u16 *ptr) {
        TV d01 = _mm512_loadu_si512((__m512i const *) ptr + 0);;
        TV d02 = _mm512_loadu_si512((__m512i const *) ptr + 1);;
        sort_02v_descending(d01, d02);
        _mm512_storeu_si512((__m512i *) ptr + 0, d01);
        _mm512_storeu_si512((__m512i *) ptr + 1, d02);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_03v_full_descending(u16 *ptr) {
        TV d01 = _mm512_loadu_si512((__m512i const *) ptr + 0);;
        TV d02 = _mm512_loadu_si512((__m512i const *) ptr + 1);;
        TV d03 = _mm512_loadu_si512((__m512i const *) ptr + 2);;
        sort_03v_descending(d01, d02, d03);
        _mm512_storeu_si512((__m512i *) ptr + 0, d01);
        _mm512_storeu_si512((__m512i *) ptr + 1, d02);
        _mm512_storeu_si512((__m512i *) ptr + 2, d03);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_04v_full_descending(u16 *ptr) {
        TV d01 = _mm512_loadu_si512((__m512i const *) ptr + 0);;
        TV d02 = _mm512_loadu_si512((__m512i const *) ptr + 1);;
        TV d03 = _mm512_loadu_si512((__m512i const *) ptr + 2);;
        TV d04 = _mm512_loadu_si512((__m512i const *) ptr + 3);;
        sort_04v_descending(d01, d02, d03, d04);
        _mm512_storeu_si512((__m512i *) ptr + 0, d01);
        _mm512_storeu_si512((__m512i *) ptr + 1, d02);
        _mm512_storeu_si512((__m512i *) ptr + 2, d03);
        _mm512_storeu_si512((__m512i *) ptr + 3, d04);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_full_vectors_ascending(u16 *ptr, usize length) {
        assert(length % N == 0);
        switch(length / N) {
            case 1: sort_01v_full_ascending(ptr); break;
            case 2: sort_02v_full_ascending(ptr); break;
            case 3: sort_03v_full_ascending(ptr); break;
            case 4: sort_04v_full_ascending(ptr); break;
        }
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_full_vectors_descending(u16 *ptr, usize length) {
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
