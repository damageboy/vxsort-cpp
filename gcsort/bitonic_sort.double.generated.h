
#include <immintrin.h>

namespace gcsort {
namespace smallsort {


static inline void bitonic_sort_double_01v_ascending(__m256d& d01) {
    __m256d  min, max, s;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    s = _mm256_shuffle_pd((__m256d) d01, (__m256d) d01, 0x5);
    
    min = _mm256_min_pd(s, d01);
    max = _mm256_max_pd(s, d01);
    d01 = _mm256_blend_pd((__m256d) min, (__m256d) max, 0xA);

    s = _mm256_permute4x64_pd((__m256d) d01, 0x1B);
    
    min = _mm256_min_pd(s, d01);
    max = _mm256_max_pd(s, d01);
    d01 = _mm256_blend_pd((__m256d) min, (__m256d) max, 0xC);

    s = _mm256_shuffle_pd((__m256d) d01, (__m256d) d01, 0x5);
    
    min = _mm256_min_pd(s, d01);
    max = _mm256_max_pd(s, d01);
    d01 = _mm256_blend_pd((__m256d) min, (__m256d) max, 0xA);
}

static inline void bitonic_sort_double_01v_merge_ascending(__m256d& d01) {
    __m256d  min, max, s;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    s = _mm256_permute4x64_pd((__m256d) d01, 0x4E);
    
    min = _mm256_min_pd(s, d01);
    max = _mm256_max_pd(s, d01);
    d01 = _mm256_blend_pd((__m256d) min, (__m256d) max, 0xC);

    s = _mm256_shuffle_pd((__m256d) d01, (__m256d) d01, 0x5);
    
    min = _mm256_min_pd(s, d01);
    max = _mm256_max_pd(s, d01);
    d01 = _mm256_blend_pd((__m256d) min, (__m256d) max, 0xA);
}

static inline void bitonic_sort_double_01v_descending(__m256d& d01) {
    __m256d  min, max, s;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    s = _mm256_shuffle_pd((__m256d) d01, (__m256d) d01, 0x5);
    
    min = _mm256_min_pd(s, d01);
    max = _mm256_max_pd(s, d01);
    d01 = _mm256_blend_pd((__m256d) max, (__m256d) min, 0xA);

    s = _mm256_permute4x64_pd((__m256d) d01, 0x1B);
    
    min = _mm256_min_pd(s, d01);
    max = _mm256_max_pd(s, d01);
    d01 = _mm256_blend_pd((__m256d) max, (__m256d) min, 0xC);

    s = _mm256_shuffle_pd((__m256d) d01, (__m256d) d01, 0x5);
    
    min = _mm256_min_pd(s, d01);
    max = _mm256_max_pd(s, d01);
    d01 = _mm256_blend_pd((__m256d) max, (__m256d) min, 0xA);
}

static inline void bitonic_sort_double_01v_merge_descending(__m256d& d01) {
    __m256d  min, max, s;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    s = _mm256_permute4x64_pd((__m256d) d01, 0x4E);
    
    min = _mm256_min_pd(s, d01);
    max = _mm256_max_pd(s, d01);
    d01 = _mm256_blend_pd((__m256d) max, (__m256d) min, 0xC);

    s = _mm256_shuffle_pd((__m256d) d01, (__m256d) d01, 0x5);
    
    min = _mm256_min_pd(s, d01);
    max = _mm256_max_pd(s, d01);
    d01 = _mm256_blend_pd((__m256d) max, (__m256d) min, 0xA);
}

static inline void bitonic_sort_double_02v_ascending(__m256d& d01, __m256d& d02) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_01v_ascending(d01);
    bitonic_sort_double_01v_descending(d02);

    tmp = d02;
    
    d02 = _mm256_max_pd(d01, d02);
    d01 = _mm256_min_pd(d01, tmp);

    bitonic_sort_double_01v_merge_ascending(d01);
    bitonic_sort_double_01v_merge_ascending(d02);
}

static inline void bitonic_sort_double_02v_descending(__m256d& d01, __m256d& d02) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_01v_descending(d01);
    bitonic_sort_double_01v_ascending(d02);

    tmp = d02;
    
    d02 = _mm256_max_pd(d01, d02);
    d01 = _mm256_min_pd(d01, tmp);

    bitonic_sort_double_01v_merge_descending(d01);
    bitonic_sort_double_01v_merge_descending(d02);
}

static inline void bitonic_sort_double_02v_merge_ascending(__m256d& d01, __m256d& d02) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d02, d01);
    
    d02 = _mm256_max_pd(d02, tmp);

    bitonic_sort_double_01v_merge_ascending(d01);
    bitonic_sort_double_01v_merge_ascending(d02);
}

static inline void bitonic_sort_double_02v_merge_descending(__m256d& d01, __m256d& d02) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d02, d01);
    
    d02 = _mm256_max_pd(d02, tmp);

    bitonic_sort_double_01v_merge_descending(d01);
    bitonic_sort_double_01v_merge_descending(d02);
}

static inline void bitonic_sort_double_03v_ascending(__m256d& d01, __m256d& d02, __m256d& d03) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_02v_ascending(d01, d02);
    bitonic_sort_double_01v_descending(d03);

    tmp = d03;
    
    d03 = _mm256_max_pd(d02, d03);
    d02 = _mm256_min_pd(d02, tmp);

    bitonic_sort_double_02v_merge_ascending(d01, d02);
    bitonic_sort_double_01v_merge_ascending(d03);
}

static inline void bitonic_sort_double_03v_descending(__m256d& d01, __m256d& d02, __m256d& d03) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_02v_descending(d01, d02);
    bitonic_sort_double_01v_ascending(d03);

    tmp = d03;
    
    d03 = _mm256_max_pd(d02, d03);
    d02 = _mm256_min_pd(d02, tmp);

    bitonic_sort_double_02v_merge_descending(d01, d02);
    bitonic_sort_double_01v_merge_descending(d03);
}

static inline void bitonic_sort_double_03v_merge_ascending(__m256d& d01, __m256d& d02, __m256d& d03) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d03, d01);
    
    d03 = _mm256_max_pd(d03, tmp);

    bitonic_sort_double_02v_merge_ascending(d01, d02);
    bitonic_sort_double_01v_merge_ascending(d03);
}

static inline void bitonic_sort_double_03v_merge_descending(__m256d& d01, __m256d& d02, __m256d& d03) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d03, d01);
    
    d03 = _mm256_max_pd(d03, tmp);

    bitonic_sort_double_02v_merge_descending(d01, d02);
    bitonic_sort_double_01v_merge_descending(d03);
}

static inline void bitonic_sort_double_04v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_02v_ascending(d01, d02);
    bitonic_sort_double_02v_descending(d03, d04);

    tmp = d03;
    
    d03 = _mm256_max_pd(d02, d03);
    d02 = _mm256_min_pd(d02, tmp);

    tmp = d04;
    
    d04 = _mm256_max_pd(d01, d04);
    d01 = _mm256_min_pd(d01, tmp);

    bitonic_sort_double_02v_merge_ascending(d01, d02);
    bitonic_sort_double_02v_merge_ascending(d03, d04);
}

static inline void bitonic_sort_double_04v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_02v_descending(d01, d02);
    bitonic_sort_double_02v_ascending(d03, d04);

    tmp = d03;
    
    d03 = _mm256_max_pd(d02, d03);
    d02 = _mm256_min_pd(d02, tmp);

    tmp = d04;
    
    d04 = _mm256_max_pd(d01, d04);
    d01 = _mm256_min_pd(d01, tmp);

    bitonic_sort_double_02v_merge_descending(d01, d02);
    bitonic_sort_double_02v_merge_descending(d03, d04);
}

static inline void bitonic_sort_double_04v_merge_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d03, d01);
    
    d03 = _mm256_max_pd(d03, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d04, d02);
    
    d04 = _mm256_max_pd(d04, tmp);

    bitonic_sort_double_02v_merge_ascending(d01, d02);
    bitonic_sort_double_02v_merge_ascending(d03, d04);
}

static inline void bitonic_sort_double_04v_merge_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d03, d01);
    
    d03 = _mm256_max_pd(d03, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d04, d02);
    
    d04 = _mm256_max_pd(d04, tmp);

    bitonic_sort_double_02v_merge_descending(d01, d02);
    bitonic_sort_double_02v_merge_descending(d03, d04);
}

static inline void bitonic_sort_double_05v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_04v_ascending(d01, d02, d03, d04);
    bitonic_sort_double_01v_descending(d05);

    tmp = d05;
    
    d05 = _mm256_max_pd(d04, d05);
    d04 = _mm256_min_pd(d04, tmp);

    bitonic_sort_double_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_double_01v_merge_ascending(d05);
}

static inline void bitonic_sort_double_05v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_04v_descending(d01, d02, d03, d04);
    bitonic_sort_double_01v_ascending(d05);

    tmp = d05;
    
    d05 = _mm256_max_pd(d04, d05);
    d04 = _mm256_min_pd(d04, tmp);

    bitonic_sort_double_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_double_01v_merge_descending(d05);
}

static inline void bitonic_sort_double_05v_merge_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d05, d01);
    
    d05 = _mm256_max_pd(d05, tmp);

    bitonic_sort_double_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_double_01v_merge_ascending(d05);
}

static inline void bitonic_sort_double_05v_merge_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d05, d01);
    
    d05 = _mm256_max_pd(d05, tmp);

    bitonic_sort_double_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_double_01v_merge_descending(d05);
}

static inline void bitonic_sort_double_06v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_04v_ascending(d01, d02, d03, d04);
    bitonic_sort_double_02v_descending(d05, d06);

    tmp = d05;
    
    d05 = _mm256_max_pd(d04, d05);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d06;
    
    d06 = _mm256_max_pd(d03, d06);
    d03 = _mm256_min_pd(d03, tmp);

    bitonic_sort_double_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_double_02v_merge_ascending(d05, d06);
}

static inline void bitonic_sort_double_06v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_04v_descending(d01, d02, d03, d04);
    bitonic_sort_double_02v_ascending(d05, d06);

    tmp = d05;
    
    d05 = _mm256_max_pd(d04, d05);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d06;
    
    d06 = _mm256_max_pd(d03, d06);
    d03 = _mm256_min_pd(d03, tmp);

    bitonic_sort_double_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_double_02v_merge_descending(d05, d06);
}

static inline void bitonic_sort_double_06v_merge_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d05, d01);
    
    d05 = _mm256_max_pd(d05, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d06, d02);
    
    d06 = _mm256_max_pd(d06, tmp);

    bitonic_sort_double_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_double_02v_merge_ascending(d05, d06);
}

static inline void bitonic_sort_double_06v_merge_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d05, d01);
    
    d05 = _mm256_max_pd(d05, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d06, d02);
    
    d06 = _mm256_max_pd(d06, tmp);

    bitonic_sort_double_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_double_02v_merge_descending(d05, d06);
}

static inline void bitonic_sort_double_07v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_04v_ascending(d01, d02, d03, d04);
    bitonic_sort_double_03v_descending(d05, d06, d07);

    tmp = d05;
    
    d05 = _mm256_max_pd(d04, d05);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d06;
    
    d06 = _mm256_max_pd(d03, d06);
    d03 = _mm256_min_pd(d03, tmp);

    tmp = d07;
    
    d07 = _mm256_max_pd(d02, d07);
    d02 = _mm256_min_pd(d02, tmp);

    bitonic_sort_double_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_double_03v_merge_ascending(d05, d06, d07);
}

static inline void bitonic_sort_double_07v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_04v_descending(d01, d02, d03, d04);
    bitonic_sort_double_03v_ascending(d05, d06, d07);

    tmp = d05;
    
    d05 = _mm256_max_pd(d04, d05);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d06;
    
    d06 = _mm256_max_pd(d03, d06);
    d03 = _mm256_min_pd(d03, tmp);

    tmp = d07;
    
    d07 = _mm256_max_pd(d02, d07);
    d02 = _mm256_min_pd(d02, tmp);

    bitonic_sort_double_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_double_03v_merge_descending(d05, d06, d07);
}

static inline void bitonic_sort_double_07v_merge_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d05, d01);
    
    d05 = _mm256_max_pd(d05, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d06, d02);
    
    d06 = _mm256_max_pd(d06, tmp);

    tmp = d03;
    
    d03 = _mm256_min_pd(d07, d03);
    
    d07 = _mm256_max_pd(d07, tmp);

    bitonic_sort_double_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_double_03v_merge_ascending(d05, d06, d07);
}

static inline void bitonic_sort_double_07v_merge_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d05, d01);
    
    d05 = _mm256_max_pd(d05, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d06, d02);
    
    d06 = _mm256_max_pd(d06, tmp);

    tmp = d03;
    
    d03 = _mm256_min_pd(d07, d03);
    
    d07 = _mm256_max_pd(d07, tmp);

    bitonic_sort_double_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_double_03v_merge_descending(d05, d06, d07);
}

static inline void bitonic_sort_double_08v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_04v_ascending(d01, d02, d03, d04);
    bitonic_sort_double_04v_descending(d05, d06, d07, d08);

    tmp = d05;
    
    d05 = _mm256_max_pd(d04, d05);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d06;
    
    d06 = _mm256_max_pd(d03, d06);
    d03 = _mm256_min_pd(d03, tmp);

    tmp = d07;
    
    d07 = _mm256_max_pd(d02, d07);
    d02 = _mm256_min_pd(d02, tmp);

    tmp = d08;
    
    d08 = _mm256_max_pd(d01, d08);
    d01 = _mm256_min_pd(d01, tmp);

    bitonic_sort_double_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_double_04v_merge_ascending(d05, d06, d07, d08);
}

static inline void bitonic_sort_double_08v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_04v_descending(d01, d02, d03, d04);
    bitonic_sort_double_04v_ascending(d05, d06, d07, d08);

    tmp = d05;
    
    d05 = _mm256_max_pd(d04, d05);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d06;
    
    d06 = _mm256_max_pd(d03, d06);
    d03 = _mm256_min_pd(d03, tmp);

    tmp = d07;
    
    d07 = _mm256_max_pd(d02, d07);
    d02 = _mm256_min_pd(d02, tmp);

    tmp = d08;
    
    d08 = _mm256_max_pd(d01, d08);
    d01 = _mm256_min_pd(d01, tmp);

    bitonic_sort_double_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_double_04v_merge_descending(d05, d06, d07, d08);
}

static inline void bitonic_sort_double_08v_merge_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d05, d01);
    
    d05 = _mm256_max_pd(d05, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d06, d02);
    
    d06 = _mm256_max_pd(d06, tmp);

    tmp = d03;
    
    d03 = _mm256_min_pd(d07, d03);
    
    d07 = _mm256_max_pd(d07, tmp);

    tmp = d04;
    
    d04 = _mm256_min_pd(d08, d04);
    
    d08 = _mm256_max_pd(d08, tmp);

    bitonic_sort_double_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_double_04v_merge_ascending(d05, d06, d07, d08);
}

static inline void bitonic_sort_double_08v_merge_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d05, d01);
    
    d05 = _mm256_max_pd(d05, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d06, d02);
    
    d06 = _mm256_max_pd(d06, tmp);

    tmp = d03;
    
    d03 = _mm256_min_pd(d07, d03);
    
    d07 = _mm256_max_pd(d07, tmp);

    tmp = d04;
    
    d04 = _mm256_min_pd(d08, d04);
    
    d08 = _mm256_max_pd(d08, tmp);

    bitonic_sort_double_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_double_04v_merge_descending(d05, d06, d07, d08);
}

static inline void bitonic_sort_double_09v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_01v_descending(d09);

    tmp = d09;
    
    d09 = _mm256_max_pd(d08, d09);
    d08 = _mm256_min_pd(d08, tmp);

    bitonic_sort_double_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_01v_merge_ascending(d09);
}

static inline void bitonic_sort_double_09v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_01v_ascending(d09);

    tmp = d09;
    
    d09 = _mm256_max_pd(d08, d09);
    d08 = _mm256_min_pd(d08, tmp);

    bitonic_sort_double_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_01v_merge_descending(d09);
}

static inline void bitonic_sort_double_09v_merge_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d09, d01);
    
    d09 = _mm256_max_pd(d09, tmp);

    bitonic_sort_double_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_01v_merge_ascending(d09);
}

static inline void bitonic_sort_double_09v_merge_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d09, d01);
    
    d09 = _mm256_max_pd(d09, tmp);

    bitonic_sort_double_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_01v_merge_descending(d09);
}

static inline void bitonic_sort_double_10v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_02v_descending(d09, d10);

    tmp = d09;
    
    d09 = _mm256_max_pd(d08, d09);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_pd(d07, d10);
    d07 = _mm256_min_pd(d07, tmp);

    bitonic_sort_double_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_02v_merge_ascending(d09, d10);
}

static inline void bitonic_sort_double_10v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_02v_ascending(d09, d10);

    tmp = d09;
    
    d09 = _mm256_max_pd(d08, d09);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_pd(d07, d10);
    d07 = _mm256_min_pd(d07, tmp);

    bitonic_sort_double_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_02v_merge_descending(d09, d10);
}

static inline void bitonic_sort_double_10v_merge_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d09, d01);
    
    d09 = _mm256_max_pd(d09, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d10, d02);
    
    d10 = _mm256_max_pd(d10, tmp);

    bitonic_sort_double_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_02v_merge_ascending(d09, d10);
}

static inline void bitonic_sort_double_10v_merge_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d09, d01);
    
    d09 = _mm256_max_pd(d09, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d10, d02);
    
    d10 = _mm256_max_pd(d10, tmp);

    bitonic_sort_double_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_02v_merge_descending(d09, d10);
}

static inline void bitonic_sort_double_11v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_03v_descending(d09, d10, d11);

    tmp = d09;
    
    d09 = _mm256_max_pd(d08, d09);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_pd(d07, d10);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_pd(d06, d11);
    d06 = _mm256_min_pd(d06, tmp);

    bitonic_sort_double_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_03v_merge_ascending(d09, d10, d11);
}

static inline void bitonic_sort_double_11v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_03v_ascending(d09, d10, d11);

    tmp = d09;
    
    d09 = _mm256_max_pd(d08, d09);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_pd(d07, d10);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_pd(d06, d11);
    d06 = _mm256_min_pd(d06, tmp);

    bitonic_sort_double_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_03v_merge_descending(d09, d10, d11);
}

static inline void bitonic_sort_double_11v_merge_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d09, d01);
    
    d09 = _mm256_max_pd(d09, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d10, d02);
    
    d10 = _mm256_max_pd(d10, tmp);

    tmp = d03;
    
    d03 = _mm256_min_pd(d11, d03);
    
    d11 = _mm256_max_pd(d11, tmp);

    bitonic_sort_double_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_03v_merge_ascending(d09, d10, d11);
}

static inline void bitonic_sort_double_11v_merge_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d09, d01);
    
    d09 = _mm256_max_pd(d09, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d10, d02);
    
    d10 = _mm256_max_pd(d10, tmp);

    tmp = d03;
    
    d03 = _mm256_min_pd(d11, d03);
    
    d11 = _mm256_max_pd(d11, tmp);

    bitonic_sort_double_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_03v_merge_descending(d09, d10, d11);
}

static inline void bitonic_sort_double_12v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_04v_descending(d09, d10, d11, d12);

    tmp = d09;
    
    d09 = _mm256_max_pd(d08, d09);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_pd(d07, d10);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_pd(d06, d11);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_pd(d05, d12);
    d05 = _mm256_min_pd(d05, tmp);

    bitonic_sort_double_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_04v_merge_ascending(d09, d10, d11, d12);
}

static inline void bitonic_sort_double_12v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_04v_ascending(d09, d10, d11, d12);

    tmp = d09;
    
    d09 = _mm256_max_pd(d08, d09);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_pd(d07, d10);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_pd(d06, d11);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_pd(d05, d12);
    d05 = _mm256_min_pd(d05, tmp);

    bitonic_sort_double_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_04v_merge_descending(d09, d10, d11, d12);
}

static inline void bitonic_sort_double_12v_merge_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d09, d01);
    
    d09 = _mm256_max_pd(d09, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d10, d02);
    
    d10 = _mm256_max_pd(d10, tmp);

    tmp = d03;
    
    d03 = _mm256_min_pd(d11, d03);
    
    d11 = _mm256_max_pd(d11, tmp);

    tmp = d04;
    
    d04 = _mm256_min_pd(d12, d04);
    
    d12 = _mm256_max_pd(d12, tmp);

    bitonic_sort_double_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_04v_merge_ascending(d09, d10, d11, d12);
}

static inline void bitonic_sort_double_12v_merge_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d09, d01);
    
    d09 = _mm256_max_pd(d09, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d10, d02);
    
    d10 = _mm256_max_pd(d10, tmp);

    tmp = d03;
    
    d03 = _mm256_min_pd(d11, d03);
    
    d11 = _mm256_max_pd(d11, tmp);

    tmp = d04;
    
    d04 = _mm256_min_pd(d12, d04);
    
    d12 = _mm256_max_pd(d12, tmp);

    bitonic_sort_double_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_04v_merge_descending(d09, d10, d11, d12);
}

static inline void bitonic_sort_double_13v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_05v_descending(d09, d10, d11, d12, d13);

    tmp = d09;
    
    d09 = _mm256_max_pd(d08, d09);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_pd(d07, d10);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_pd(d06, d11);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_pd(d05, d12);
    d05 = _mm256_min_pd(d05, tmp);

    tmp = d13;
    
    d13 = _mm256_max_pd(d04, d13);
    d04 = _mm256_min_pd(d04, tmp);

    bitonic_sort_double_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_05v_merge_ascending(d09, d10, d11, d12, d13);
}

static inline void bitonic_sort_double_13v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_05v_ascending(d09, d10, d11, d12, d13);

    tmp = d09;
    
    d09 = _mm256_max_pd(d08, d09);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_pd(d07, d10);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_pd(d06, d11);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_pd(d05, d12);
    d05 = _mm256_min_pd(d05, tmp);

    tmp = d13;
    
    d13 = _mm256_max_pd(d04, d13);
    d04 = _mm256_min_pd(d04, tmp);

    bitonic_sort_double_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_05v_merge_descending(d09, d10, d11, d12, d13);
}

static inline void bitonic_sort_double_13v_merge_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d09, d01);
    
    d09 = _mm256_max_pd(d09, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d10, d02);
    
    d10 = _mm256_max_pd(d10, tmp);

    tmp = d03;
    
    d03 = _mm256_min_pd(d11, d03);
    
    d11 = _mm256_max_pd(d11, tmp);

    tmp = d04;
    
    d04 = _mm256_min_pd(d12, d04);
    
    d12 = _mm256_max_pd(d12, tmp);

    tmp = d05;
    
    d05 = _mm256_min_pd(d13, d05);
    
    d13 = _mm256_max_pd(d13, tmp);

    bitonic_sort_double_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_05v_merge_ascending(d09, d10, d11, d12, d13);
}

static inline void bitonic_sort_double_13v_merge_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d09, d01);
    
    d09 = _mm256_max_pd(d09, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d10, d02);
    
    d10 = _mm256_max_pd(d10, tmp);

    tmp = d03;
    
    d03 = _mm256_min_pd(d11, d03);
    
    d11 = _mm256_max_pd(d11, tmp);

    tmp = d04;
    
    d04 = _mm256_min_pd(d12, d04);
    
    d12 = _mm256_max_pd(d12, tmp);

    tmp = d05;
    
    d05 = _mm256_min_pd(d13, d05);
    
    d13 = _mm256_max_pd(d13, tmp);

    bitonic_sort_double_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_05v_merge_descending(d09, d10, d11, d12, d13);
}

static inline void bitonic_sort_double_14v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_06v_descending(d09, d10, d11, d12, d13, d14);

    tmp = d09;
    
    d09 = _mm256_max_pd(d08, d09);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_pd(d07, d10);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_pd(d06, d11);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_pd(d05, d12);
    d05 = _mm256_min_pd(d05, tmp);

    tmp = d13;
    
    d13 = _mm256_max_pd(d04, d13);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d14;
    
    d14 = _mm256_max_pd(d03, d14);
    d03 = _mm256_min_pd(d03, tmp);

    bitonic_sort_double_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_06v_merge_ascending(d09, d10, d11, d12, d13, d14);
}

static inline void bitonic_sort_double_14v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_06v_ascending(d09, d10, d11, d12, d13, d14);

    tmp = d09;
    
    d09 = _mm256_max_pd(d08, d09);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_pd(d07, d10);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_pd(d06, d11);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_pd(d05, d12);
    d05 = _mm256_min_pd(d05, tmp);

    tmp = d13;
    
    d13 = _mm256_max_pd(d04, d13);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d14;
    
    d14 = _mm256_max_pd(d03, d14);
    d03 = _mm256_min_pd(d03, tmp);

    bitonic_sort_double_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_06v_merge_descending(d09, d10, d11, d12, d13, d14);
}

static inline void bitonic_sort_double_14v_merge_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d09, d01);
    
    d09 = _mm256_max_pd(d09, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d10, d02);
    
    d10 = _mm256_max_pd(d10, tmp);

    tmp = d03;
    
    d03 = _mm256_min_pd(d11, d03);
    
    d11 = _mm256_max_pd(d11, tmp);

    tmp = d04;
    
    d04 = _mm256_min_pd(d12, d04);
    
    d12 = _mm256_max_pd(d12, tmp);

    tmp = d05;
    
    d05 = _mm256_min_pd(d13, d05);
    
    d13 = _mm256_max_pd(d13, tmp);

    tmp = d06;
    
    d06 = _mm256_min_pd(d14, d06);
    
    d14 = _mm256_max_pd(d14, tmp);

    bitonic_sort_double_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_06v_merge_ascending(d09, d10, d11, d12, d13, d14);
}

static inline void bitonic_sort_double_14v_merge_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d09, d01);
    
    d09 = _mm256_max_pd(d09, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d10, d02);
    
    d10 = _mm256_max_pd(d10, tmp);

    tmp = d03;
    
    d03 = _mm256_min_pd(d11, d03);
    
    d11 = _mm256_max_pd(d11, tmp);

    tmp = d04;
    
    d04 = _mm256_min_pd(d12, d04);
    
    d12 = _mm256_max_pd(d12, tmp);

    tmp = d05;
    
    d05 = _mm256_min_pd(d13, d05);
    
    d13 = _mm256_max_pd(d13, tmp);

    tmp = d06;
    
    d06 = _mm256_min_pd(d14, d06);
    
    d14 = _mm256_max_pd(d14, tmp);

    bitonic_sort_double_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_06v_merge_descending(d09, d10, d11, d12, d13, d14);
}

static inline void bitonic_sort_double_15v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_07v_descending(d09, d10, d11, d12, d13, d14, d15);

    tmp = d09;
    
    d09 = _mm256_max_pd(d08, d09);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_pd(d07, d10);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_pd(d06, d11);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_pd(d05, d12);
    d05 = _mm256_min_pd(d05, tmp);

    tmp = d13;
    
    d13 = _mm256_max_pd(d04, d13);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d14;
    
    d14 = _mm256_max_pd(d03, d14);
    d03 = _mm256_min_pd(d03, tmp);

    tmp = d15;
    
    d15 = _mm256_max_pd(d02, d15);
    d02 = _mm256_min_pd(d02, tmp);

    bitonic_sort_double_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_07v_merge_ascending(d09, d10, d11, d12, d13, d14, d15);
}

static inline void bitonic_sort_double_15v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_07v_ascending(d09, d10, d11, d12, d13, d14, d15);

    tmp = d09;
    
    d09 = _mm256_max_pd(d08, d09);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_pd(d07, d10);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_pd(d06, d11);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_pd(d05, d12);
    d05 = _mm256_min_pd(d05, tmp);

    tmp = d13;
    
    d13 = _mm256_max_pd(d04, d13);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d14;
    
    d14 = _mm256_max_pd(d03, d14);
    d03 = _mm256_min_pd(d03, tmp);

    tmp = d15;
    
    d15 = _mm256_max_pd(d02, d15);
    d02 = _mm256_min_pd(d02, tmp);

    bitonic_sort_double_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_07v_merge_descending(d09, d10, d11, d12, d13, d14, d15);
}

static inline void bitonic_sort_double_15v_merge_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d09, d01);
    
    d09 = _mm256_max_pd(d09, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d10, d02);
    
    d10 = _mm256_max_pd(d10, tmp);

    tmp = d03;
    
    d03 = _mm256_min_pd(d11, d03);
    
    d11 = _mm256_max_pd(d11, tmp);

    tmp = d04;
    
    d04 = _mm256_min_pd(d12, d04);
    
    d12 = _mm256_max_pd(d12, tmp);

    tmp = d05;
    
    d05 = _mm256_min_pd(d13, d05);
    
    d13 = _mm256_max_pd(d13, tmp);

    tmp = d06;
    
    d06 = _mm256_min_pd(d14, d06);
    
    d14 = _mm256_max_pd(d14, tmp);

    tmp = d07;
    
    d07 = _mm256_min_pd(d15, d07);
    
    d15 = _mm256_max_pd(d15, tmp);

    bitonic_sort_double_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_07v_merge_ascending(d09, d10, d11, d12, d13, d14, d15);
}

static inline void bitonic_sort_double_15v_merge_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d09, d01);
    
    d09 = _mm256_max_pd(d09, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d10, d02);
    
    d10 = _mm256_max_pd(d10, tmp);

    tmp = d03;
    
    d03 = _mm256_min_pd(d11, d03);
    
    d11 = _mm256_max_pd(d11, tmp);

    tmp = d04;
    
    d04 = _mm256_min_pd(d12, d04);
    
    d12 = _mm256_max_pd(d12, tmp);

    tmp = d05;
    
    d05 = _mm256_min_pd(d13, d05);
    
    d13 = _mm256_max_pd(d13, tmp);

    tmp = d06;
    
    d06 = _mm256_min_pd(d14, d06);
    
    d14 = _mm256_max_pd(d14, tmp);

    tmp = d07;
    
    d07 = _mm256_min_pd(d15, d07);
    
    d15 = _mm256_max_pd(d15, tmp);

    bitonic_sort_double_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_07v_merge_descending(d09, d10, d11, d12, d13, d14, d15);
}

static inline void bitonic_sort_double_16v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_08v_descending(d09, d10, d11, d12, d13, d14, d15, d16);

    tmp = d09;
    
    d09 = _mm256_max_pd(d08, d09);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_pd(d07, d10);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_pd(d06, d11);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_pd(d05, d12);
    d05 = _mm256_min_pd(d05, tmp);

    tmp = d13;
    
    d13 = _mm256_max_pd(d04, d13);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d14;
    
    d14 = _mm256_max_pd(d03, d14);
    d03 = _mm256_min_pd(d03, tmp);

    tmp = d15;
    
    d15 = _mm256_max_pd(d02, d15);
    d02 = _mm256_min_pd(d02, tmp);

    tmp = d16;
    
    d16 = _mm256_max_pd(d01, d16);
    d01 = _mm256_min_pd(d01, tmp);

    bitonic_sort_double_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_08v_merge_ascending(d09, d10, d11, d12, d13, d14, d15, d16);
}

static inline void bitonic_sort_double_16v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_08v_ascending(d09, d10, d11, d12, d13, d14, d15, d16);

    tmp = d09;
    
    d09 = _mm256_max_pd(d08, d09);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_pd(d07, d10);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_pd(d06, d11);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_pd(d05, d12);
    d05 = _mm256_min_pd(d05, tmp);

    tmp = d13;
    
    d13 = _mm256_max_pd(d04, d13);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d14;
    
    d14 = _mm256_max_pd(d03, d14);
    d03 = _mm256_min_pd(d03, tmp);

    tmp = d15;
    
    d15 = _mm256_max_pd(d02, d15);
    d02 = _mm256_min_pd(d02, tmp);

    tmp = d16;
    
    d16 = _mm256_max_pd(d01, d16);
    d01 = _mm256_min_pd(d01, tmp);

    bitonic_sort_double_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_08v_merge_descending(d09, d10, d11, d12, d13, d14, d15, d16);
}

static inline void bitonic_sort_double_16v_merge_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d09, d01);
    
    d09 = _mm256_max_pd(d09, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d10, d02);
    
    d10 = _mm256_max_pd(d10, tmp);

    tmp = d03;
    
    d03 = _mm256_min_pd(d11, d03);
    
    d11 = _mm256_max_pd(d11, tmp);

    tmp = d04;
    
    d04 = _mm256_min_pd(d12, d04);
    
    d12 = _mm256_max_pd(d12, tmp);

    tmp = d05;
    
    d05 = _mm256_min_pd(d13, d05);
    
    d13 = _mm256_max_pd(d13, tmp);

    tmp = d06;
    
    d06 = _mm256_min_pd(d14, d06);
    
    d14 = _mm256_max_pd(d14, tmp);

    tmp = d07;
    
    d07 = _mm256_min_pd(d15, d07);
    
    d15 = _mm256_max_pd(d15, tmp);

    tmp = d08;
    
    d08 = _mm256_min_pd(d16, d08);
    
    d16 = _mm256_max_pd(d16, tmp);

    bitonic_sort_double_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_08v_merge_ascending(d09, d10, d11, d12, d13, d14, d15, d16);
}

static inline void bitonic_sort_double_16v_merge_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_pd(d09, d01);
    
    d09 = _mm256_max_pd(d09, tmp);

    tmp = d02;
    
    d02 = _mm256_min_pd(d10, d02);
    
    d10 = _mm256_max_pd(d10, tmp);

    tmp = d03;
    
    d03 = _mm256_min_pd(d11, d03);
    
    d11 = _mm256_max_pd(d11, tmp);

    tmp = d04;
    
    d04 = _mm256_min_pd(d12, d04);
    
    d12 = _mm256_max_pd(d12, tmp);

    tmp = d05;
    
    d05 = _mm256_min_pd(d13, d05);
    
    d13 = _mm256_max_pd(d13, tmp);

    tmp = d06;
    
    d06 = _mm256_min_pd(d14, d06);
    
    d14 = _mm256_max_pd(d14, tmp);

    tmp = d07;
    
    d07 = _mm256_min_pd(d15, d07);
    
    d15 = _mm256_max_pd(d15, tmp);

    tmp = d08;
    
    d08 = _mm256_min_pd(d16, d08);
    
    d16 = _mm256_max_pd(d16, tmp);

    bitonic_sort_double_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_double_08v_merge_descending(d09, d10, d11, d12, d13, d14, d15, d16);
}

static inline void bitonic_sort_double_17v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_01v_descending(d17);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    bitonic_sort_double_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_01v_merge_ascending(d17);
}

static inline void bitonic_sort_double_17v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_01v_ascending(d17);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    bitonic_sort_double_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_01v_merge_descending(d17);
}

static inline void bitonic_sort_double_18v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_02v_descending(d17, d18);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    bitonic_sort_double_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_02v_merge_ascending(d17, d18);
}

static inline void bitonic_sort_double_18v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_02v_ascending(d17, d18);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    bitonic_sort_double_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_02v_merge_descending(d17, d18);
}

static inline void bitonic_sort_double_19v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_03v_descending(d17, d18, d19);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    bitonic_sort_double_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_03v_merge_ascending(d17, d18, d19);
}

static inline void bitonic_sort_double_19v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_03v_ascending(d17, d18, d19);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    bitonic_sort_double_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_03v_merge_descending(d17, d18, d19);
}

static inline void bitonic_sort_double_20v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_04v_descending(d17, d18, d19, d20);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    bitonic_sort_double_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_04v_merge_ascending(d17, d18, d19, d20);
}

static inline void bitonic_sort_double_20v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_04v_ascending(d17, d18, d19, d20);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    bitonic_sort_double_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_04v_merge_descending(d17, d18, d19, d20);
}

static inline void bitonic_sort_double_21v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_05v_descending(d17, d18, d19, d20, d21);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    bitonic_sort_double_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_05v_merge_ascending(d17, d18, d19, d20, d21);
}

static inline void bitonic_sort_double_21v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_05v_ascending(d17, d18, d19, d20, d21);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    bitonic_sort_double_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_05v_merge_descending(d17, d18, d19, d20, d21);
}

static inline void bitonic_sort_double_22v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_06v_descending(d17, d18, d19, d20, d21, d22);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    bitonic_sort_double_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_06v_merge_ascending(d17, d18, d19, d20, d21, d22);
}

static inline void bitonic_sort_double_22v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_06v_ascending(d17, d18, d19, d20, d21, d22);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    bitonic_sort_double_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_06v_merge_descending(d17, d18, d19, d20, d21, d22);
}

static inline void bitonic_sort_double_23v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_07v_descending(d17, d18, d19, d20, d21, d22, d23);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    bitonic_sort_double_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_07v_merge_ascending(d17, d18, d19, d20, d21, d22, d23);
}

static inline void bitonic_sort_double_23v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_07v_ascending(d17, d18, d19, d20, d21, d22, d23);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    bitonic_sort_double_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_07v_merge_descending(d17, d18, d19, d20, d21, d22, d23);
}

static inline void bitonic_sort_double_24v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_08v_descending(d17, d18, d19, d20, d21, d22, d23, d24);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    bitonic_sort_double_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_08v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24);
}

static inline void bitonic_sort_double_24v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_08v_ascending(d17, d18, d19, d20, d21, d22, d23, d24);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    bitonic_sort_double_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_08v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24);
}

static inline void bitonic_sort_double_25v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24, __m256d& d25) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_09v_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    tmp = d25;
    
    d25 = _mm256_max_pd(d08, d25);
    d08 = _mm256_min_pd(d08, tmp);

    bitonic_sort_double_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_09v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25);
}

static inline void bitonic_sort_double_25v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24, __m256d& d25) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_09v_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    tmp = d25;
    
    d25 = _mm256_max_pd(d08, d25);
    d08 = _mm256_min_pd(d08, tmp);

    bitonic_sort_double_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_09v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25);
}

static inline void bitonic_sort_double_26v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24, __m256d& d25, __m256d& d26) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_10v_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    tmp = d25;
    
    d25 = _mm256_max_pd(d08, d25);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d26;
    
    d26 = _mm256_max_pd(d07, d26);
    d07 = _mm256_min_pd(d07, tmp);

    bitonic_sort_double_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_10v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26);
}

static inline void bitonic_sort_double_26v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24, __m256d& d25, __m256d& d26) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_10v_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    tmp = d25;
    
    d25 = _mm256_max_pd(d08, d25);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d26;
    
    d26 = _mm256_max_pd(d07, d26);
    d07 = _mm256_min_pd(d07, tmp);

    bitonic_sort_double_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_10v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26);
}

static inline void bitonic_sort_double_27v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24, __m256d& d25, __m256d& d26, __m256d& d27) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_11v_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    tmp = d25;
    
    d25 = _mm256_max_pd(d08, d25);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d26;
    
    d26 = _mm256_max_pd(d07, d26);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d27;
    
    d27 = _mm256_max_pd(d06, d27);
    d06 = _mm256_min_pd(d06, tmp);

    bitonic_sort_double_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_11v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27);
}

static inline void bitonic_sort_double_27v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24, __m256d& d25, __m256d& d26, __m256d& d27) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_11v_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    tmp = d25;
    
    d25 = _mm256_max_pd(d08, d25);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d26;
    
    d26 = _mm256_max_pd(d07, d26);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d27;
    
    d27 = _mm256_max_pd(d06, d27);
    d06 = _mm256_min_pd(d06, tmp);

    bitonic_sort_double_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_11v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27);
}

static inline void bitonic_sort_double_28v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24, __m256d& d25, __m256d& d26, __m256d& d27, __m256d& d28) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_12v_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    tmp = d25;
    
    d25 = _mm256_max_pd(d08, d25);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d26;
    
    d26 = _mm256_max_pd(d07, d26);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d27;
    
    d27 = _mm256_max_pd(d06, d27);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d28;
    
    d28 = _mm256_max_pd(d05, d28);
    d05 = _mm256_min_pd(d05, tmp);

    bitonic_sort_double_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_12v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28);
}

static inline void bitonic_sort_double_28v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24, __m256d& d25, __m256d& d26, __m256d& d27, __m256d& d28) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_12v_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    tmp = d25;
    
    d25 = _mm256_max_pd(d08, d25);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d26;
    
    d26 = _mm256_max_pd(d07, d26);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d27;
    
    d27 = _mm256_max_pd(d06, d27);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d28;
    
    d28 = _mm256_max_pd(d05, d28);
    d05 = _mm256_min_pd(d05, tmp);

    bitonic_sort_double_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_12v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28);
}

static inline void bitonic_sort_double_29v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24, __m256d& d25, __m256d& d26, __m256d& d27, __m256d& d28, __m256d& d29) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_13v_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    tmp = d25;
    
    d25 = _mm256_max_pd(d08, d25);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d26;
    
    d26 = _mm256_max_pd(d07, d26);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d27;
    
    d27 = _mm256_max_pd(d06, d27);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d28;
    
    d28 = _mm256_max_pd(d05, d28);
    d05 = _mm256_min_pd(d05, tmp);

    tmp = d29;
    
    d29 = _mm256_max_pd(d04, d29);
    d04 = _mm256_min_pd(d04, tmp);

    bitonic_sort_double_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_13v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29);
}

static inline void bitonic_sort_double_29v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24, __m256d& d25, __m256d& d26, __m256d& d27, __m256d& d28, __m256d& d29) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_13v_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    tmp = d25;
    
    d25 = _mm256_max_pd(d08, d25);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d26;
    
    d26 = _mm256_max_pd(d07, d26);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d27;
    
    d27 = _mm256_max_pd(d06, d27);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d28;
    
    d28 = _mm256_max_pd(d05, d28);
    d05 = _mm256_min_pd(d05, tmp);

    tmp = d29;
    
    d29 = _mm256_max_pd(d04, d29);
    d04 = _mm256_min_pd(d04, tmp);

    bitonic_sort_double_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_13v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29);
}

static inline void bitonic_sort_double_30v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24, __m256d& d25, __m256d& d26, __m256d& d27, __m256d& d28, __m256d& d29, __m256d& d30) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_14v_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    tmp = d25;
    
    d25 = _mm256_max_pd(d08, d25);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d26;
    
    d26 = _mm256_max_pd(d07, d26);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d27;
    
    d27 = _mm256_max_pd(d06, d27);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d28;
    
    d28 = _mm256_max_pd(d05, d28);
    d05 = _mm256_min_pd(d05, tmp);

    tmp = d29;
    
    d29 = _mm256_max_pd(d04, d29);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d30;
    
    d30 = _mm256_max_pd(d03, d30);
    d03 = _mm256_min_pd(d03, tmp);

    bitonic_sort_double_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_14v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30);
}

static inline void bitonic_sort_double_30v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24, __m256d& d25, __m256d& d26, __m256d& d27, __m256d& d28, __m256d& d29, __m256d& d30) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_14v_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    tmp = d25;
    
    d25 = _mm256_max_pd(d08, d25);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d26;
    
    d26 = _mm256_max_pd(d07, d26);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d27;
    
    d27 = _mm256_max_pd(d06, d27);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d28;
    
    d28 = _mm256_max_pd(d05, d28);
    d05 = _mm256_min_pd(d05, tmp);

    tmp = d29;
    
    d29 = _mm256_max_pd(d04, d29);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d30;
    
    d30 = _mm256_max_pd(d03, d30);
    d03 = _mm256_min_pd(d03, tmp);

    bitonic_sort_double_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_14v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30);
}

static inline void bitonic_sort_double_31v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24, __m256d& d25, __m256d& d26, __m256d& d27, __m256d& d28, __m256d& d29, __m256d& d30, __m256d& d31) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_15v_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    tmp = d25;
    
    d25 = _mm256_max_pd(d08, d25);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d26;
    
    d26 = _mm256_max_pd(d07, d26);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d27;
    
    d27 = _mm256_max_pd(d06, d27);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d28;
    
    d28 = _mm256_max_pd(d05, d28);
    d05 = _mm256_min_pd(d05, tmp);

    tmp = d29;
    
    d29 = _mm256_max_pd(d04, d29);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d30;
    
    d30 = _mm256_max_pd(d03, d30);
    d03 = _mm256_min_pd(d03, tmp);

    tmp = d31;
    
    d31 = _mm256_max_pd(d02, d31);
    d02 = _mm256_min_pd(d02, tmp);

    bitonic_sort_double_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_15v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31);
}

static inline void bitonic_sort_double_31v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24, __m256d& d25, __m256d& d26, __m256d& d27, __m256d& d28, __m256d& d29, __m256d& d30, __m256d& d31) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_15v_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    tmp = d25;
    
    d25 = _mm256_max_pd(d08, d25);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d26;
    
    d26 = _mm256_max_pd(d07, d26);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d27;
    
    d27 = _mm256_max_pd(d06, d27);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d28;
    
    d28 = _mm256_max_pd(d05, d28);
    d05 = _mm256_min_pd(d05, tmp);

    tmp = d29;
    
    d29 = _mm256_max_pd(d04, d29);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d30;
    
    d30 = _mm256_max_pd(d03, d30);
    d03 = _mm256_min_pd(d03, tmp);

    tmp = d31;
    
    d31 = _mm256_max_pd(d02, d31);
    d02 = _mm256_min_pd(d02, tmp);

    bitonic_sort_double_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_15v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31);
}

static inline void bitonic_sort_double_32v_ascending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24, __m256d& d25, __m256d& d26, __m256d& d27, __m256d& d28, __m256d& d29, __m256d& d30, __m256d& d31, __m256d& d32) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_16v_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    tmp = d25;
    
    d25 = _mm256_max_pd(d08, d25);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d26;
    
    d26 = _mm256_max_pd(d07, d26);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d27;
    
    d27 = _mm256_max_pd(d06, d27);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d28;
    
    d28 = _mm256_max_pd(d05, d28);
    d05 = _mm256_min_pd(d05, tmp);

    tmp = d29;
    
    d29 = _mm256_max_pd(d04, d29);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d30;
    
    d30 = _mm256_max_pd(d03, d30);
    d03 = _mm256_min_pd(d03, tmp);

    tmp = d31;
    
    d31 = _mm256_max_pd(d02, d31);
    d02 = _mm256_min_pd(d02, tmp);

    tmp = d32;
    
    d32 = _mm256_max_pd(d01, d32);
    d01 = _mm256_min_pd(d01, tmp);

    bitonic_sort_double_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_16v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32);
}

static inline void bitonic_sort_double_32v_descending(__m256d& d01, __m256d& d02, __m256d& d03, __m256d& d04, __m256d& d05, __m256d& d06, __m256d& d07, __m256d& d08, __m256d& d09, __m256d& d10, __m256d& d11, __m256d& d12, __m256d& d13, __m256d& d14, __m256d& d15, __m256d& d16, __m256d& d17, __m256d& d18, __m256d& d19, __m256d& d20, __m256d& d21, __m256d& d22, __m256d& d23, __m256d& d24, __m256d& d25, __m256d& d26, __m256d& d27, __m256d& d28, __m256d& d29, __m256d& d30, __m256d& d31, __m256d& d32) {
    __m256d  tmp;
    __m256d topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_double_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_16v_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32);

    tmp = d17;
    
    d17 = _mm256_max_pd(d16, d17);
    d16 = _mm256_min_pd(d16, tmp);

    tmp = d18;
    
    d18 = _mm256_max_pd(d15, d18);
    d15 = _mm256_min_pd(d15, tmp);

    tmp = d19;
    
    d19 = _mm256_max_pd(d14, d19);
    d14 = _mm256_min_pd(d14, tmp);

    tmp = d20;
    
    d20 = _mm256_max_pd(d13, d20);
    d13 = _mm256_min_pd(d13, tmp);

    tmp = d21;
    
    d21 = _mm256_max_pd(d12, d21);
    d12 = _mm256_min_pd(d12, tmp);

    tmp = d22;
    
    d22 = _mm256_max_pd(d11, d22);
    d11 = _mm256_min_pd(d11, tmp);

    tmp = d23;
    
    d23 = _mm256_max_pd(d10, d23);
    d10 = _mm256_min_pd(d10, tmp);

    tmp = d24;
    
    d24 = _mm256_max_pd(d09, d24);
    d09 = _mm256_min_pd(d09, tmp);

    tmp = d25;
    
    d25 = _mm256_max_pd(d08, d25);
    d08 = _mm256_min_pd(d08, tmp);

    tmp = d26;
    
    d26 = _mm256_max_pd(d07, d26);
    d07 = _mm256_min_pd(d07, tmp);

    tmp = d27;
    
    d27 = _mm256_max_pd(d06, d27);
    d06 = _mm256_min_pd(d06, tmp);

    tmp = d28;
    
    d28 = _mm256_max_pd(d05, d28);
    d05 = _mm256_min_pd(d05, tmp);

    tmp = d29;
    
    d29 = _mm256_max_pd(d04, d29);
    d04 = _mm256_min_pd(d04, tmp);

    tmp = d30;
    
    d30 = _mm256_max_pd(d03, d30);
    d03 = _mm256_min_pd(d03, tmp);

    tmp = d31;
    
    d31 = _mm256_max_pd(d02, d31);
    d02 = _mm256_min_pd(d02, tmp);

    tmp = d32;
    
    d32 = _mm256_max_pd(d01, d32);
    d01 = _mm256_min_pd(d01, tmp);

    bitonic_sort_double_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_double_16v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32);
}

static __attribute__((noinline)) void bitonic_sort_double_01v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    bitonic_sort_double_01v_ascending(d01);
    _mm256_storeu_pd((double *) ptr + 0, d01);
}

static __attribute__((noinline)) void bitonic_sort_double_02v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    bitonic_sort_double_02v_ascending(d01, d02);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
}

static __attribute__((noinline)) void bitonic_sort_double_03v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    bitonic_sort_double_03v_ascending(d01, d02, d03);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
}

static __attribute__((noinline)) void bitonic_sort_double_04v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    bitonic_sort_double_04v_ascending(d01, d02, d03, d04);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
}

static __attribute__((noinline)) void bitonic_sort_double_05v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    bitonic_sort_double_05v_ascending(d01, d02, d03, d04, d05);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
}

static __attribute__((noinline)) void bitonic_sort_double_06v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    bitonic_sort_double_06v_ascending(d01, d02, d03, d04, d05, d06);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
}

static __attribute__((noinline)) void bitonic_sort_double_07v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    bitonic_sort_double_07v_ascending(d01, d02, d03, d04, d05, d06, d07);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
}

static __attribute__((noinline)) void bitonic_sort_double_08v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    bitonic_sort_double_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
}

static __attribute__((noinline)) void bitonic_sort_double_09v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    bitonic_sort_double_09v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
}

static __attribute__((noinline)) void bitonic_sort_double_10v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    bitonic_sort_double_10v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
}

static __attribute__((noinline)) void bitonic_sort_double_11v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    bitonic_sort_double_11v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
}

static __attribute__((noinline)) void bitonic_sort_double_12v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    bitonic_sort_double_12v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
}

static __attribute__((noinline)) void bitonic_sort_double_13v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    bitonic_sort_double_13v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
}

static __attribute__((noinline)) void bitonic_sort_double_14v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    bitonic_sort_double_14v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
}

static __attribute__((noinline)) void bitonic_sort_double_15v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    bitonic_sort_double_15v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
}

static __attribute__((noinline)) void bitonic_sort_double_16v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    bitonic_sort_double_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
}

static __attribute__((noinline)) void bitonic_sort_double_17v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    __m256d d17 = _mm256_loadu_pd((double const *) ptr + 16);
    bitonic_sort_double_17v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
    _mm256_storeu_pd((double *) ptr + 16, d17);
}

static __attribute__((noinline)) void bitonic_sort_double_18v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    __m256d d17 = _mm256_loadu_pd((double const *) ptr + 16);
    __m256d d18 = _mm256_loadu_pd((double const *) ptr + 17);
    bitonic_sort_double_18v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
    _mm256_storeu_pd((double *) ptr + 16, d17);
    _mm256_storeu_pd((double *) ptr + 17, d18);
}

static __attribute__((noinline)) void bitonic_sort_double_19v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    __m256d d17 = _mm256_loadu_pd((double const *) ptr + 16);
    __m256d d18 = _mm256_loadu_pd((double const *) ptr + 17);
    __m256d d19 = _mm256_loadu_pd((double const *) ptr + 18);
    bitonic_sort_double_19v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
    _mm256_storeu_pd((double *) ptr + 16, d17);
    _mm256_storeu_pd((double *) ptr + 17, d18);
    _mm256_storeu_pd((double *) ptr + 18, d19);
}

static __attribute__((noinline)) void bitonic_sort_double_20v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    __m256d d17 = _mm256_loadu_pd((double const *) ptr + 16);
    __m256d d18 = _mm256_loadu_pd((double const *) ptr + 17);
    __m256d d19 = _mm256_loadu_pd((double const *) ptr + 18);
    __m256d d20 = _mm256_loadu_pd((double const *) ptr + 19);
    bitonic_sort_double_20v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
    _mm256_storeu_pd((double *) ptr + 16, d17);
    _mm256_storeu_pd((double *) ptr + 17, d18);
    _mm256_storeu_pd((double *) ptr + 18, d19);
    _mm256_storeu_pd((double *) ptr + 19, d20);
}

static __attribute__((noinline)) void bitonic_sort_double_21v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    __m256d d17 = _mm256_loadu_pd((double const *) ptr + 16);
    __m256d d18 = _mm256_loadu_pd((double const *) ptr + 17);
    __m256d d19 = _mm256_loadu_pd((double const *) ptr + 18);
    __m256d d20 = _mm256_loadu_pd((double const *) ptr + 19);
    __m256d d21 = _mm256_loadu_pd((double const *) ptr + 20);
    bitonic_sort_double_21v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
    _mm256_storeu_pd((double *) ptr + 16, d17);
    _mm256_storeu_pd((double *) ptr + 17, d18);
    _mm256_storeu_pd((double *) ptr + 18, d19);
    _mm256_storeu_pd((double *) ptr + 19, d20);
    _mm256_storeu_pd((double *) ptr + 20, d21);
}

static __attribute__((noinline)) void bitonic_sort_double_22v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    __m256d d17 = _mm256_loadu_pd((double const *) ptr + 16);
    __m256d d18 = _mm256_loadu_pd((double const *) ptr + 17);
    __m256d d19 = _mm256_loadu_pd((double const *) ptr + 18);
    __m256d d20 = _mm256_loadu_pd((double const *) ptr + 19);
    __m256d d21 = _mm256_loadu_pd((double const *) ptr + 20);
    __m256d d22 = _mm256_loadu_pd((double const *) ptr + 21);
    bitonic_sort_double_22v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
    _mm256_storeu_pd((double *) ptr + 16, d17);
    _mm256_storeu_pd((double *) ptr + 17, d18);
    _mm256_storeu_pd((double *) ptr + 18, d19);
    _mm256_storeu_pd((double *) ptr + 19, d20);
    _mm256_storeu_pd((double *) ptr + 20, d21);
    _mm256_storeu_pd((double *) ptr + 21, d22);
}

static __attribute__((noinline)) void bitonic_sort_double_23v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    __m256d d17 = _mm256_loadu_pd((double const *) ptr + 16);
    __m256d d18 = _mm256_loadu_pd((double const *) ptr + 17);
    __m256d d19 = _mm256_loadu_pd((double const *) ptr + 18);
    __m256d d20 = _mm256_loadu_pd((double const *) ptr + 19);
    __m256d d21 = _mm256_loadu_pd((double const *) ptr + 20);
    __m256d d22 = _mm256_loadu_pd((double const *) ptr + 21);
    __m256d d23 = _mm256_loadu_pd((double const *) ptr + 22);
    bitonic_sort_double_23v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
    _mm256_storeu_pd((double *) ptr + 16, d17);
    _mm256_storeu_pd((double *) ptr + 17, d18);
    _mm256_storeu_pd((double *) ptr + 18, d19);
    _mm256_storeu_pd((double *) ptr + 19, d20);
    _mm256_storeu_pd((double *) ptr + 20, d21);
    _mm256_storeu_pd((double *) ptr + 21, d22);
    _mm256_storeu_pd((double *) ptr + 22, d23);
}

static __attribute__((noinline)) void bitonic_sort_double_24v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    __m256d d17 = _mm256_loadu_pd((double const *) ptr + 16);
    __m256d d18 = _mm256_loadu_pd((double const *) ptr + 17);
    __m256d d19 = _mm256_loadu_pd((double const *) ptr + 18);
    __m256d d20 = _mm256_loadu_pd((double const *) ptr + 19);
    __m256d d21 = _mm256_loadu_pd((double const *) ptr + 20);
    __m256d d22 = _mm256_loadu_pd((double const *) ptr + 21);
    __m256d d23 = _mm256_loadu_pd((double const *) ptr + 22);
    __m256d d24 = _mm256_loadu_pd((double const *) ptr + 23);
    bitonic_sort_double_24v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
    _mm256_storeu_pd((double *) ptr + 16, d17);
    _mm256_storeu_pd((double *) ptr + 17, d18);
    _mm256_storeu_pd((double *) ptr + 18, d19);
    _mm256_storeu_pd((double *) ptr + 19, d20);
    _mm256_storeu_pd((double *) ptr + 20, d21);
    _mm256_storeu_pd((double *) ptr + 21, d22);
    _mm256_storeu_pd((double *) ptr + 22, d23);
    _mm256_storeu_pd((double *) ptr + 23, d24);
}

static __attribute__((noinline)) void bitonic_sort_double_25v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    __m256d d17 = _mm256_loadu_pd((double const *) ptr + 16);
    __m256d d18 = _mm256_loadu_pd((double const *) ptr + 17);
    __m256d d19 = _mm256_loadu_pd((double const *) ptr + 18);
    __m256d d20 = _mm256_loadu_pd((double const *) ptr + 19);
    __m256d d21 = _mm256_loadu_pd((double const *) ptr + 20);
    __m256d d22 = _mm256_loadu_pd((double const *) ptr + 21);
    __m256d d23 = _mm256_loadu_pd((double const *) ptr + 22);
    __m256d d24 = _mm256_loadu_pd((double const *) ptr + 23);
    __m256d d25 = _mm256_loadu_pd((double const *) ptr + 24);
    bitonic_sort_double_25v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
    _mm256_storeu_pd((double *) ptr + 16, d17);
    _mm256_storeu_pd((double *) ptr + 17, d18);
    _mm256_storeu_pd((double *) ptr + 18, d19);
    _mm256_storeu_pd((double *) ptr + 19, d20);
    _mm256_storeu_pd((double *) ptr + 20, d21);
    _mm256_storeu_pd((double *) ptr + 21, d22);
    _mm256_storeu_pd((double *) ptr + 22, d23);
    _mm256_storeu_pd((double *) ptr + 23, d24);
    _mm256_storeu_pd((double *) ptr + 24, d25);
}

static __attribute__((noinline)) void bitonic_sort_double_26v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    __m256d d17 = _mm256_loadu_pd((double const *) ptr + 16);
    __m256d d18 = _mm256_loadu_pd((double const *) ptr + 17);
    __m256d d19 = _mm256_loadu_pd((double const *) ptr + 18);
    __m256d d20 = _mm256_loadu_pd((double const *) ptr + 19);
    __m256d d21 = _mm256_loadu_pd((double const *) ptr + 20);
    __m256d d22 = _mm256_loadu_pd((double const *) ptr + 21);
    __m256d d23 = _mm256_loadu_pd((double const *) ptr + 22);
    __m256d d24 = _mm256_loadu_pd((double const *) ptr + 23);
    __m256d d25 = _mm256_loadu_pd((double const *) ptr + 24);
    __m256d d26 = _mm256_loadu_pd((double const *) ptr + 25);
    bitonic_sort_double_26v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
    _mm256_storeu_pd((double *) ptr + 16, d17);
    _mm256_storeu_pd((double *) ptr + 17, d18);
    _mm256_storeu_pd((double *) ptr + 18, d19);
    _mm256_storeu_pd((double *) ptr + 19, d20);
    _mm256_storeu_pd((double *) ptr + 20, d21);
    _mm256_storeu_pd((double *) ptr + 21, d22);
    _mm256_storeu_pd((double *) ptr + 22, d23);
    _mm256_storeu_pd((double *) ptr + 23, d24);
    _mm256_storeu_pd((double *) ptr + 24, d25);
    _mm256_storeu_pd((double *) ptr + 25, d26);
}

static __attribute__((noinline)) void bitonic_sort_double_27v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    __m256d d17 = _mm256_loadu_pd((double const *) ptr + 16);
    __m256d d18 = _mm256_loadu_pd((double const *) ptr + 17);
    __m256d d19 = _mm256_loadu_pd((double const *) ptr + 18);
    __m256d d20 = _mm256_loadu_pd((double const *) ptr + 19);
    __m256d d21 = _mm256_loadu_pd((double const *) ptr + 20);
    __m256d d22 = _mm256_loadu_pd((double const *) ptr + 21);
    __m256d d23 = _mm256_loadu_pd((double const *) ptr + 22);
    __m256d d24 = _mm256_loadu_pd((double const *) ptr + 23);
    __m256d d25 = _mm256_loadu_pd((double const *) ptr + 24);
    __m256d d26 = _mm256_loadu_pd((double const *) ptr + 25);
    __m256d d27 = _mm256_loadu_pd((double const *) ptr + 26);
    bitonic_sort_double_27v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
    _mm256_storeu_pd((double *) ptr + 16, d17);
    _mm256_storeu_pd((double *) ptr + 17, d18);
    _mm256_storeu_pd((double *) ptr + 18, d19);
    _mm256_storeu_pd((double *) ptr + 19, d20);
    _mm256_storeu_pd((double *) ptr + 20, d21);
    _mm256_storeu_pd((double *) ptr + 21, d22);
    _mm256_storeu_pd((double *) ptr + 22, d23);
    _mm256_storeu_pd((double *) ptr + 23, d24);
    _mm256_storeu_pd((double *) ptr + 24, d25);
    _mm256_storeu_pd((double *) ptr + 25, d26);
    _mm256_storeu_pd((double *) ptr + 26, d27);
}

static __attribute__((noinline)) void bitonic_sort_double_28v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    __m256d d17 = _mm256_loadu_pd((double const *) ptr + 16);
    __m256d d18 = _mm256_loadu_pd((double const *) ptr + 17);
    __m256d d19 = _mm256_loadu_pd((double const *) ptr + 18);
    __m256d d20 = _mm256_loadu_pd((double const *) ptr + 19);
    __m256d d21 = _mm256_loadu_pd((double const *) ptr + 20);
    __m256d d22 = _mm256_loadu_pd((double const *) ptr + 21);
    __m256d d23 = _mm256_loadu_pd((double const *) ptr + 22);
    __m256d d24 = _mm256_loadu_pd((double const *) ptr + 23);
    __m256d d25 = _mm256_loadu_pd((double const *) ptr + 24);
    __m256d d26 = _mm256_loadu_pd((double const *) ptr + 25);
    __m256d d27 = _mm256_loadu_pd((double const *) ptr + 26);
    __m256d d28 = _mm256_loadu_pd((double const *) ptr + 27);
    bitonic_sort_double_28v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
    _mm256_storeu_pd((double *) ptr + 16, d17);
    _mm256_storeu_pd((double *) ptr + 17, d18);
    _mm256_storeu_pd((double *) ptr + 18, d19);
    _mm256_storeu_pd((double *) ptr + 19, d20);
    _mm256_storeu_pd((double *) ptr + 20, d21);
    _mm256_storeu_pd((double *) ptr + 21, d22);
    _mm256_storeu_pd((double *) ptr + 22, d23);
    _mm256_storeu_pd((double *) ptr + 23, d24);
    _mm256_storeu_pd((double *) ptr + 24, d25);
    _mm256_storeu_pd((double *) ptr + 25, d26);
    _mm256_storeu_pd((double *) ptr + 26, d27);
    _mm256_storeu_pd((double *) ptr + 27, d28);
}

static __attribute__((noinline)) void bitonic_sort_double_29v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    __m256d d17 = _mm256_loadu_pd((double const *) ptr + 16);
    __m256d d18 = _mm256_loadu_pd((double const *) ptr + 17);
    __m256d d19 = _mm256_loadu_pd((double const *) ptr + 18);
    __m256d d20 = _mm256_loadu_pd((double const *) ptr + 19);
    __m256d d21 = _mm256_loadu_pd((double const *) ptr + 20);
    __m256d d22 = _mm256_loadu_pd((double const *) ptr + 21);
    __m256d d23 = _mm256_loadu_pd((double const *) ptr + 22);
    __m256d d24 = _mm256_loadu_pd((double const *) ptr + 23);
    __m256d d25 = _mm256_loadu_pd((double const *) ptr + 24);
    __m256d d26 = _mm256_loadu_pd((double const *) ptr + 25);
    __m256d d27 = _mm256_loadu_pd((double const *) ptr + 26);
    __m256d d28 = _mm256_loadu_pd((double const *) ptr + 27);
    __m256d d29 = _mm256_loadu_pd((double const *) ptr + 28);
    bitonic_sort_double_29v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
    _mm256_storeu_pd((double *) ptr + 16, d17);
    _mm256_storeu_pd((double *) ptr + 17, d18);
    _mm256_storeu_pd((double *) ptr + 18, d19);
    _mm256_storeu_pd((double *) ptr + 19, d20);
    _mm256_storeu_pd((double *) ptr + 20, d21);
    _mm256_storeu_pd((double *) ptr + 21, d22);
    _mm256_storeu_pd((double *) ptr + 22, d23);
    _mm256_storeu_pd((double *) ptr + 23, d24);
    _mm256_storeu_pd((double *) ptr + 24, d25);
    _mm256_storeu_pd((double *) ptr + 25, d26);
    _mm256_storeu_pd((double *) ptr + 26, d27);
    _mm256_storeu_pd((double *) ptr + 27, d28);
    _mm256_storeu_pd((double *) ptr + 28, d29);
}

static __attribute__((noinline)) void bitonic_sort_double_30v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    __m256d d17 = _mm256_loadu_pd((double const *) ptr + 16);
    __m256d d18 = _mm256_loadu_pd((double const *) ptr + 17);
    __m256d d19 = _mm256_loadu_pd((double const *) ptr + 18);
    __m256d d20 = _mm256_loadu_pd((double const *) ptr + 19);
    __m256d d21 = _mm256_loadu_pd((double const *) ptr + 20);
    __m256d d22 = _mm256_loadu_pd((double const *) ptr + 21);
    __m256d d23 = _mm256_loadu_pd((double const *) ptr + 22);
    __m256d d24 = _mm256_loadu_pd((double const *) ptr + 23);
    __m256d d25 = _mm256_loadu_pd((double const *) ptr + 24);
    __m256d d26 = _mm256_loadu_pd((double const *) ptr + 25);
    __m256d d27 = _mm256_loadu_pd((double const *) ptr + 26);
    __m256d d28 = _mm256_loadu_pd((double const *) ptr + 27);
    __m256d d29 = _mm256_loadu_pd((double const *) ptr + 28);
    __m256d d30 = _mm256_loadu_pd((double const *) ptr + 29);
    bitonic_sort_double_30v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
    _mm256_storeu_pd((double *) ptr + 16, d17);
    _mm256_storeu_pd((double *) ptr + 17, d18);
    _mm256_storeu_pd((double *) ptr + 18, d19);
    _mm256_storeu_pd((double *) ptr + 19, d20);
    _mm256_storeu_pd((double *) ptr + 20, d21);
    _mm256_storeu_pd((double *) ptr + 21, d22);
    _mm256_storeu_pd((double *) ptr + 22, d23);
    _mm256_storeu_pd((double *) ptr + 23, d24);
    _mm256_storeu_pd((double *) ptr + 24, d25);
    _mm256_storeu_pd((double *) ptr + 25, d26);
    _mm256_storeu_pd((double *) ptr + 26, d27);
    _mm256_storeu_pd((double *) ptr + 27, d28);
    _mm256_storeu_pd((double *) ptr + 28, d29);
    _mm256_storeu_pd((double *) ptr + 29, d30);
}

static __attribute__((noinline)) void bitonic_sort_double_31v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    __m256d d17 = _mm256_loadu_pd((double const *) ptr + 16);
    __m256d d18 = _mm256_loadu_pd((double const *) ptr + 17);
    __m256d d19 = _mm256_loadu_pd((double const *) ptr + 18);
    __m256d d20 = _mm256_loadu_pd((double const *) ptr + 19);
    __m256d d21 = _mm256_loadu_pd((double const *) ptr + 20);
    __m256d d22 = _mm256_loadu_pd((double const *) ptr + 21);
    __m256d d23 = _mm256_loadu_pd((double const *) ptr + 22);
    __m256d d24 = _mm256_loadu_pd((double const *) ptr + 23);
    __m256d d25 = _mm256_loadu_pd((double const *) ptr + 24);
    __m256d d26 = _mm256_loadu_pd((double const *) ptr + 25);
    __m256d d27 = _mm256_loadu_pd((double const *) ptr + 26);
    __m256d d28 = _mm256_loadu_pd((double const *) ptr + 27);
    __m256d d29 = _mm256_loadu_pd((double const *) ptr + 28);
    __m256d d30 = _mm256_loadu_pd((double const *) ptr + 29);
    __m256d d31 = _mm256_loadu_pd((double const *) ptr + 30);
    bitonic_sort_double_31v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
    _mm256_storeu_pd((double *) ptr + 16, d17);
    _mm256_storeu_pd((double *) ptr + 17, d18);
    _mm256_storeu_pd((double *) ptr + 18, d19);
    _mm256_storeu_pd((double *) ptr + 19, d20);
    _mm256_storeu_pd((double *) ptr + 20, d21);
    _mm256_storeu_pd((double *) ptr + 21, d22);
    _mm256_storeu_pd((double *) ptr + 22, d23);
    _mm256_storeu_pd((double *) ptr + 23, d24);
    _mm256_storeu_pd((double *) ptr + 24, d25);
    _mm256_storeu_pd((double *) ptr + 25, d26);
    _mm256_storeu_pd((double *) ptr + 26, d27);
    _mm256_storeu_pd((double *) ptr + 27, d28);
    _mm256_storeu_pd((double *) ptr + 28, d29);
    _mm256_storeu_pd((double *) ptr + 29, d30);
    _mm256_storeu_pd((double *) ptr + 30, d31);
}

static __attribute__((noinline)) void bitonic_sort_double_32v(double *ptr) {
    __m256d d01 = _mm256_loadu_pd((double const *) ptr + 0);
    __m256d d02 = _mm256_loadu_pd((double const *) ptr + 1);
    __m256d d03 = _mm256_loadu_pd((double const *) ptr + 2);
    __m256d d04 = _mm256_loadu_pd((double const *) ptr + 3);
    __m256d d05 = _mm256_loadu_pd((double const *) ptr + 4);
    __m256d d06 = _mm256_loadu_pd((double const *) ptr + 5);
    __m256d d07 = _mm256_loadu_pd((double const *) ptr + 6);
    __m256d d08 = _mm256_loadu_pd((double const *) ptr + 7);
    __m256d d09 = _mm256_loadu_pd((double const *) ptr + 8);
    __m256d d10 = _mm256_loadu_pd((double const *) ptr + 9);
    __m256d d11 = _mm256_loadu_pd((double const *) ptr + 10);
    __m256d d12 = _mm256_loadu_pd((double const *) ptr + 11);
    __m256d d13 = _mm256_loadu_pd((double const *) ptr + 12);
    __m256d d14 = _mm256_loadu_pd((double const *) ptr + 13);
    __m256d d15 = _mm256_loadu_pd((double const *) ptr + 14);
    __m256d d16 = _mm256_loadu_pd((double const *) ptr + 15);
    __m256d d17 = _mm256_loadu_pd((double const *) ptr + 16);
    __m256d d18 = _mm256_loadu_pd((double const *) ptr + 17);
    __m256d d19 = _mm256_loadu_pd((double const *) ptr + 18);
    __m256d d20 = _mm256_loadu_pd((double const *) ptr + 19);
    __m256d d21 = _mm256_loadu_pd((double const *) ptr + 20);
    __m256d d22 = _mm256_loadu_pd((double const *) ptr + 21);
    __m256d d23 = _mm256_loadu_pd((double const *) ptr + 22);
    __m256d d24 = _mm256_loadu_pd((double const *) ptr + 23);
    __m256d d25 = _mm256_loadu_pd((double const *) ptr + 24);
    __m256d d26 = _mm256_loadu_pd((double const *) ptr + 25);
    __m256d d27 = _mm256_loadu_pd((double const *) ptr + 26);
    __m256d d28 = _mm256_loadu_pd((double const *) ptr + 27);
    __m256d d29 = _mm256_loadu_pd((double const *) ptr + 28);
    __m256d d30 = _mm256_loadu_pd((double const *) ptr + 29);
    __m256d d31 = _mm256_loadu_pd((double const *) ptr + 30);
    __m256d d32 = _mm256_loadu_pd((double const *) ptr + 31);
    bitonic_sort_double_32v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32);
    _mm256_storeu_pd((double *) ptr + 0, d01);
    _mm256_storeu_pd((double *) ptr + 1, d02);
    _mm256_storeu_pd((double *) ptr + 2, d03);
    _mm256_storeu_pd((double *) ptr + 3, d04);
    _mm256_storeu_pd((double *) ptr + 4, d05);
    _mm256_storeu_pd((double *) ptr + 5, d06);
    _mm256_storeu_pd((double *) ptr + 6, d07);
    _mm256_storeu_pd((double *) ptr + 7, d08);
    _mm256_storeu_pd((double *) ptr + 8, d09);
    _mm256_storeu_pd((double *) ptr + 9, d10);
    _mm256_storeu_pd((double *) ptr + 10, d11);
    _mm256_storeu_pd((double *) ptr + 11, d12);
    _mm256_storeu_pd((double *) ptr + 12, d13);
    _mm256_storeu_pd((double *) ptr + 13, d14);
    _mm256_storeu_pd((double *) ptr + 14, d15);
    _mm256_storeu_pd((double *) ptr + 15, d16);
    _mm256_storeu_pd((double *) ptr + 16, d17);
    _mm256_storeu_pd((double *) ptr + 17, d18);
    _mm256_storeu_pd((double *) ptr + 18, d19);
    _mm256_storeu_pd((double *) ptr + 19, d20);
    _mm256_storeu_pd((double *) ptr + 20, d21);
    _mm256_storeu_pd((double *) ptr + 21, d22);
    _mm256_storeu_pd((double *) ptr + 22, d23);
    _mm256_storeu_pd((double *) ptr + 23, d24);
    _mm256_storeu_pd((double *) ptr + 24, d25);
    _mm256_storeu_pd((double *) ptr + 25, d26);
    _mm256_storeu_pd((double *) ptr + 26, d27);
    _mm256_storeu_pd((double *) ptr + 27, d28);
    _mm256_storeu_pd((double *) ptr + 28, d29);
    _mm256_storeu_pd((double *) ptr + 29, d30);
    _mm256_storeu_pd((double *) ptr + 30, d31);
    _mm256_storeu_pd((double *) ptr + 31, d32);
}

static void bitonic_sort_double(double *ptr, int length) {
    const int N = 4;

    switch(length / N) {
        case 1: bitonic_sort_double_01v(ptr); break;
        case 2: bitonic_sort_double_02v(ptr); break;
        case 3: bitonic_sort_double_03v(ptr); break;
        case 4: bitonic_sort_double_04v(ptr); break;
        case 5: bitonic_sort_double_05v(ptr); break;
        case 6: bitonic_sort_double_06v(ptr); break;
        case 7: bitonic_sort_double_07v(ptr); break;
        case 8: bitonic_sort_double_08v(ptr); break;
        case 9: bitonic_sort_double_09v(ptr); break;
        case 10: bitonic_sort_double_10v(ptr); break;
        case 11: bitonic_sort_double_11v(ptr); break;
        case 12: bitonic_sort_double_12v(ptr); break;
        case 13: bitonic_sort_double_13v(ptr); break;
        case 14: bitonic_sort_double_14v(ptr); break;
        case 15: bitonic_sort_double_15v(ptr); break;
        case 16: bitonic_sort_double_16v(ptr); break;
        case 17: bitonic_sort_double_17v(ptr); break;
        case 18: bitonic_sort_double_18v(ptr); break;
        case 19: bitonic_sort_double_19v(ptr); break;
        case 20: bitonic_sort_double_20v(ptr); break;
        case 21: bitonic_sort_double_21v(ptr); break;
        case 22: bitonic_sort_double_22v(ptr); break;
        case 23: bitonic_sort_double_23v(ptr); break;
        case 24: bitonic_sort_double_24v(ptr); break;
        case 25: bitonic_sort_double_25v(ptr); break;
        case 26: bitonic_sort_double_26v(ptr); break;
        case 27: bitonic_sort_double_27v(ptr); break;
        case 28: bitonic_sort_double_28v(ptr); break;
        case 29: bitonic_sort_double_29v(ptr); break;
        case 30: bitonic_sort_double_30v(ptr); break;
        case 31: bitonic_sort_double_31v(ptr); break;
        case 32: bitonic_sort_double_32v(ptr); break;
    }
}
}
}
