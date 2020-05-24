
#include <immintrin.h>

namespace gcsort {
namespace smallsort {


static inline void bitonic_sort_uint64_t_01v_ascending(__m256i& d01) {
    __m256i  min, max, s, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    s = _mm256_shuffle_pd((__m256d) d01, (__m256d) d01, 0x5);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
    min = _mm256_blendv_pd(s, d01, cmp);
    max = _mm256_blendv_pd(d01, s, cmp);
    d01 = _mm256_blend_pd((__m256d) min, (__m256d) max, 0xA);

    s = _mm256_permute4x64_pd((__m256d) d01, 0x1B);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
    min = _mm256_blendv_pd(s, d01, cmp);
    max = _mm256_blendv_pd(d01, s, cmp);
    d01 = _mm256_blend_pd((__m256d) min, (__m256d) max, 0xC);

    s = _mm256_shuffle_pd((__m256d) d01, (__m256d) d01, 0x5);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
    min = _mm256_blendv_pd(s, d01, cmp);
    max = _mm256_blendv_pd(d01, s, cmp);
    d01 = _mm256_blend_pd((__m256d) min, (__m256d) max, 0xA);
}

static inline void bitonic_sort_uint64_t_01v_merge_ascending(__m256i& d01) {
    __m256i  min, max, s, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    s = _mm256_permute4x64_pd((__m256d) d01, 0x4E);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
    min = _mm256_blendv_pd(s, d01, cmp);
    max = _mm256_blendv_pd(d01, s, cmp);
    d01 = _mm256_blend_pd((__m256d) min, (__m256d) max, 0xC);

    s = _mm256_shuffle_pd((__m256d) d01, (__m256d) d01, 0x5);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
    min = _mm256_blendv_pd(s, d01, cmp);
    max = _mm256_blendv_pd(d01, s, cmp);
    d01 = _mm256_blend_pd((__m256d) min, (__m256d) max, 0xA);
}

static inline void bitonic_sort_uint64_t_01v_descending(__m256i& d01) {
    __m256i  min, max, s, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    s = _mm256_shuffle_pd((__m256d) d01, (__m256d) d01, 0x5);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
    min = _mm256_blendv_pd(s, d01, cmp);
    max = _mm256_blendv_pd(d01, s, cmp);
    d01 = _mm256_blend_pd((__m256d) max, (__m256d) min, 0xA);

    s = _mm256_permute4x64_pd((__m256d) d01, 0x1B);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
    min = _mm256_blendv_pd(s, d01, cmp);
    max = _mm256_blendv_pd(d01, s, cmp);
    d01 = _mm256_blend_pd((__m256d) max, (__m256d) min, 0xC);

    s = _mm256_shuffle_pd((__m256d) d01, (__m256d) d01, 0x5);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
    min = _mm256_blendv_pd(s, d01, cmp);
    max = _mm256_blendv_pd(d01, s, cmp);
    d01 = _mm256_blend_pd((__m256d) max, (__m256d) min, 0xA);
}

static inline void bitonic_sort_uint64_t_01v_merge_descending(__m256i& d01) {
    __m256i  min, max, s, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    s = _mm256_permute4x64_pd((__m256d) d01, 0x4E);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
    min = _mm256_blendv_pd(s, d01, cmp);
    max = _mm256_blendv_pd(d01, s, cmp);
    d01 = _mm256_blend_pd((__m256d) max, (__m256d) min, 0xC);

    s = _mm256_shuffle_pd((__m256d) d01, (__m256d) d01, 0x5);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, s), _mm256_xor_si256(topBit, d01));
    min = _mm256_blendv_pd(s, d01, cmp);
    max = _mm256_blendv_pd(d01, s, cmp);
    d01 = _mm256_blend_pd((__m256d) max, (__m256d) min, 0xA);
}

static inline void bitonic_sort_uint64_t_02v_ascending(__m256i& d01, __m256i& d02) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_01v_ascending(d01);
    bitonic_sort_uint64_t_01v_descending(d02);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d01), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d02, d01, cmp);
    d01 = _mm256_blendv_pd(d01, tmp, cmp);

    bitonic_sort_uint64_t_01v_merge_ascending(d01);
    bitonic_sort_uint64_t_01v_merge_ascending(d02);
}

static inline void bitonic_sort_uint64_t_02v_descending(__m256i& d01, __m256i& d02) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_01v_descending(d01);
    bitonic_sort_uint64_t_01v_ascending(d02);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d01), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d02, d01, cmp);
    d01 = _mm256_blendv_pd(d01, tmp, cmp);

    bitonic_sort_uint64_t_01v_merge_descending(d01);
    bitonic_sort_uint64_t_01v_merge_descending(d02);
}

static inline void bitonic_sort_uint64_t_02v_merge_ascending(__m256i& d01, __m256i& d02) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d02, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, tmp));
    d02 = _mm256_blendv_pd(tmp, d02, cmp);

    bitonic_sort_uint64_t_01v_merge_ascending(d01);
    bitonic_sort_uint64_t_01v_merge_ascending(d02);
}

static inline void bitonic_sort_uint64_t_02v_merge_descending(__m256i& d01, __m256i& d02) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d02, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, tmp));
    d02 = _mm256_blendv_pd(tmp, d02, cmp);

    bitonic_sort_uint64_t_01v_merge_descending(d01);
    bitonic_sort_uint64_t_01v_merge_descending(d02);
}

static inline void bitonic_sort_uint64_t_03v_ascending(__m256i& d01, __m256i& d02, __m256i& d03) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_02v_ascending(d01, d02);
    bitonic_sort_uint64_t_01v_descending(d03);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d03, d02, cmp);
    d02 = _mm256_blendv_pd(d02, tmp, cmp);

    bitonic_sort_uint64_t_02v_merge_ascending(d01, d02);
    bitonic_sort_uint64_t_01v_merge_ascending(d03);
}

static inline void bitonic_sort_uint64_t_03v_descending(__m256i& d01, __m256i& d02, __m256i& d03) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_02v_descending(d01, d02);
    bitonic_sort_uint64_t_01v_ascending(d03);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d03, d02, cmp);
    d02 = _mm256_blendv_pd(d02, tmp, cmp);

    bitonic_sort_uint64_t_02v_merge_descending(d01, d02);
    bitonic_sort_uint64_t_01v_merge_descending(d03);
}

static inline void bitonic_sort_uint64_t_03v_merge_ascending(__m256i& d01, __m256i& d02, __m256i& d03) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d03, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, tmp));
    d03 = _mm256_blendv_pd(tmp, d03, cmp);

    bitonic_sort_uint64_t_02v_merge_ascending(d01, d02);
    bitonic_sort_uint64_t_01v_merge_ascending(d03);
}

static inline void bitonic_sort_uint64_t_03v_merge_descending(__m256i& d01, __m256i& d02, __m256i& d03) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d03, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, tmp));
    d03 = _mm256_blendv_pd(tmp, d03, cmp);

    bitonic_sort_uint64_t_02v_merge_descending(d01, d02);
    bitonic_sort_uint64_t_01v_merge_descending(d03);
}

static inline void bitonic_sort_uint64_t_04v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_02v_ascending(d01, d02);
    bitonic_sort_uint64_t_02v_descending(d03, d04);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d03, d02, cmp);
    d02 = _mm256_blendv_pd(d02, tmp, cmp);

    tmp = d04;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d01), _mm256_xor_si256(topBit, d04));
    d04 = _mm256_blendv_pd(d04, d01, cmp);
    d01 = _mm256_blendv_pd(d01, tmp, cmp);

    bitonic_sort_uint64_t_02v_merge_ascending(d01, d02);
    bitonic_sort_uint64_t_02v_merge_ascending(d03, d04);
}

static inline void bitonic_sort_uint64_t_04v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_02v_descending(d01, d02);
    bitonic_sort_uint64_t_02v_ascending(d03, d04);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d03, d02, cmp);
    d02 = _mm256_blendv_pd(d02, tmp, cmp);

    tmp = d04;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d01), _mm256_xor_si256(topBit, d04));
    d04 = _mm256_blendv_pd(d04, d01, cmp);
    d01 = _mm256_blendv_pd(d01, tmp, cmp);

    bitonic_sort_uint64_t_02v_merge_descending(d01, d02);
    bitonic_sort_uint64_t_02v_merge_descending(d03, d04);
}

static inline void bitonic_sort_uint64_t_04v_merge_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d03, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, tmp));
    d03 = _mm256_blendv_pd(tmp, d03, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d04, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, tmp));
    d04 = _mm256_blendv_pd(tmp, d04, cmp);

    bitonic_sort_uint64_t_02v_merge_ascending(d01, d02);
    bitonic_sort_uint64_t_02v_merge_ascending(d03, d04);
}

static inline void bitonic_sort_uint64_t_04v_merge_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d03, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, tmp));
    d03 = _mm256_blendv_pd(tmp, d03, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d04, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, tmp));
    d04 = _mm256_blendv_pd(tmp, d04, cmp);

    bitonic_sort_uint64_t_02v_merge_descending(d01, d02);
    bitonic_sort_uint64_t_02v_merge_descending(d03, d04);
}

static inline void bitonic_sort_uint64_t_05v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_04v_ascending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_01v_descending(d05);

    tmp = d05;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d05));
    d05 = _mm256_blendv_pd(d05, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    bitonic_sort_uint64_t_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_01v_merge_ascending(d05);
}

static inline void bitonic_sort_uint64_t_05v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_04v_descending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_01v_ascending(d05);

    tmp = d05;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d05));
    d05 = _mm256_blendv_pd(d05, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    bitonic_sort_uint64_t_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_01v_merge_descending(d05);
}

static inline void bitonic_sort_uint64_t_05v_merge_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d05, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, tmp));
    d05 = _mm256_blendv_pd(tmp, d05, cmp);

    bitonic_sort_uint64_t_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_01v_merge_ascending(d05);
}

static inline void bitonic_sort_uint64_t_05v_merge_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d05, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, tmp));
    d05 = _mm256_blendv_pd(tmp, d05, cmp);

    bitonic_sort_uint64_t_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_01v_merge_descending(d05);
}

static inline void bitonic_sort_uint64_t_06v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_04v_ascending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_02v_descending(d05, d06);

    tmp = d05;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d05));
    d05 = _mm256_blendv_pd(d05, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d06;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d06));
    d06 = _mm256_blendv_pd(d06, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    bitonic_sort_uint64_t_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_02v_merge_ascending(d05, d06);
}

static inline void bitonic_sort_uint64_t_06v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_04v_descending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_02v_ascending(d05, d06);

    tmp = d05;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d05));
    d05 = _mm256_blendv_pd(d05, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d06;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d06));
    d06 = _mm256_blendv_pd(d06, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    bitonic_sort_uint64_t_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_02v_merge_descending(d05, d06);
}

static inline void bitonic_sort_uint64_t_06v_merge_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d05, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, tmp));
    d05 = _mm256_blendv_pd(tmp, d05, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d06, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, tmp));
    d06 = _mm256_blendv_pd(tmp, d06, cmp);

    bitonic_sort_uint64_t_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_02v_merge_ascending(d05, d06);
}

static inline void bitonic_sort_uint64_t_06v_merge_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d05, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, tmp));
    d05 = _mm256_blendv_pd(tmp, d05, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d06, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, tmp));
    d06 = _mm256_blendv_pd(tmp, d06, cmp);

    bitonic_sort_uint64_t_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_02v_merge_descending(d05, d06);
}

static inline void bitonic_sort_uint64_t_07v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_04v_ascending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_03v_descending(d05, d06, d07);

    tmp = d05;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d05));
    d05 = _mm256_blendv_pd(d05, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d06;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d06));
    d06 = _mm256_blendv_pd(d06, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    tmp = d07;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d07));
    d07 = _mm256_blendv_pd(d07, d02, cmp);
    d02 = _mm256_blendv_pd(d02, tmp, cmp);

    bitonic_sort_uint64_t_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_03v_merge_ascending(d05, d06, d07);
}

static inline void bitonic_sort_uint64_t_07v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_04v_descending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_03v_ascending(d05, d06, d07);

    tmp = d05;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d05));
    d05 = _mm256_blendv_pd(d05, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d06;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d06));
    d06 = _mm256_blendv_pd(d06, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    tmp = d07;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d07));
    d07 = _mm256_blendv_pd(d07, d02, cmp);
    d02 = _mm256_blendv_pd(d02, tmp, cmp);

    bitonic_sort_uint64_t_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_03v_merge_descending(d05, d06, d07);
}

static inline void bitonic_sort_uint64_t_07v_merge_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d05, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, tmp));
    d05 = _mm256_blendv_pd(tmp, d05, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d06, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, tmp));
    d06 = _mm256_blendv_pd(tmp, d06, cmp);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d07, d03, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, tmp));
    d07 = _mm256_blendv_pd(tmp, d07, cmp);

    bitonic_sort_uint64_t_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_03v_merge_ascending(d05, d06, d07);
}

static inline void bitonic_sort_uint64_t_07v_merge_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d05, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, tmp));
    d05 = _mm256_blendv_pd(tmp, d05, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d06, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, tmp));
    d06 = _mm256_blendv_pd(tmp, d06, cmp);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d07, d03, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, tmp));
    d07 = _mm256_blendv_pd(tmp, d07, cmp);

    bitonic_sort_uint64_t_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_03v_merge_descending(d05, d06, d07);
}

static inline void bitonic_sort_uint64_t_08v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_04v_ascending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_04v_descending(d05, d06, d07, d08);

    tmp = d05;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d05));
    d05 = _mm256_blendv_pd(d05, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d06;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d06));
    d06 = _mm256_blendv_pd(d06, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    tmp = d07;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d07));
    d07 = _mm256_blendv_pd(d07, d02, cmp);
    d02 = _mm256_blendv_pd(d02, tmp, cmp);

    tmp = d08;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d01), _mm256_xor_si256(topBit, d08));
    d08 = _mm256_blendv_pd(d08, d01, cmp);
    d01 = _mm256_blendv_pd(d01, tmp, cmp);

    bitonic_sort_uint64_t_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_04v_merge_ascending(d05, d06, d07, d08);
}

static inline void bitonic_sort_uint64_t_08v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_04v_descending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_04v_ascending(d05, d06, d07, d08);

    tmp = d05;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d05));
    d05 = _mm256_blendv_pd(d05, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d06;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d06));
    d06 = _mm256_blendv_pd(d06, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    tmp = d07;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d07));
    d07 = _mm256_blendv_pd(d07, d02, cmp);
    d02 = _mm256_blendv_pd(d02, tmp, cmp);

    tmp = d08;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d01), _mm256_xor_si256(topBit, d08));
    d08 = _mm256_blendv_pd(d08, d01, cmp);
    d01 = _mm256_blendv_pd(d01, tmp, cmp);

    bitonic_sort_uint64_t_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_04v_merge_descending(d05, d06, d07, d08);
}

static inline void bitonic_sort_uint64_t_08v_merge_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d05, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, tmp));
    d05 = _mm256_blendv_pd(tmp, d05, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d06, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, tmp));
    d06 = _mm256_blendv_pd(tmp, d06, cmp);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d07, d03, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, tmp));
    d07 = _mm256_blendv_pd(tmp, d07, cmp);

    tmp = d04;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d04));
    d04 = _mm256_blendv_pd(d08, d04, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, tmp));
    d08 = _mm256_blendv_pd(tmp, d08, cmp);

    bitonic_sort_uint64_t_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_04v_merge_ascending(d05, d06, d07, d08);
}

static inline void bitonic_sort_uint64_t_08v_merge_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d05, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, tmp));
    d05 = _mm256_blendv_pd(tmp, d05, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d06, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, tmp));
    d06 = _mm256_blendv_pd(tmp, d06, cmp);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d07, d03, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, tmp));
    d07 = _mm256_blendv_pd(tmp, d07, cmp);

    tmp = d04;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d04));
    d04 = _mm256_blendv_pd(d08, d04, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, tmp));
    d08 = _mm256_blendv_pd(tmp, d08, cmp);

    bitonic_sort_uint64_t_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_uint64_t_04v_merge_descending(d05, d06, d07, d08);
}

static inline void bitonic_sort_uint64_t_09v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_01v_descending(d09);

    tmp = d09;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d09));
    d09 = _mm256_blendv_pd(d09, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    bitonic_sort_uint64_t_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_01v_merge_ascending(d09);
}

static inline void bitonic_sort_uint64_t_09v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_01v_ascending(d09);

    tmp = d09;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d09));
    d09 = _mm256_blendv_pd(d09, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    bitonic_sort_uint64_t_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_01v_merge_descending(d09);
}

static inline void bitonic_sort_uint64_t_09v_merge_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d09, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, tmp));
    d09 = _mm256_blendv_pd(tmp, d09, cmp);

    bitonic_sort_uint64_t_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_01v_merge_ascending(d09);
}

static inline void bitonic_sort_uint64_t_09v_merge_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d09, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, tmp));
    d09 = _mm256_blendv_pd(tmp, d09, cmp);

    bitonic_sort_uint64_t_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_01v_merge_descending(d09);
}

static inline void bitonic_sort_uint64_t_10v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_02v_descending(d09, d10);

    tmp = d09;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d09));
    d09 = _mm256_blendv_pd(d09, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d10;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d10));
    d10 = _mm256_blendv_pd(d10, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    bitonic_sort_uint64_t_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_02v_merge_ascending(d09, d10);
}

static inline void bitonic_sort_uint64_t_10v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_02v_ascending(d09, d10);

    tmp = d09;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d09));
    d09 = _mm256_blendv_pd(d09, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d10;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d10));
    d10 = _mm256_blendv_pd(d10, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    bitonic_sort_uint64_t_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_02v_merge_descending(d09, d10);
}

static inline void bitonic_sort_uint64_t_10v_merge_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d09, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, tmp));
    d09 = _mm256_blendv_pd(tmp, d09, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d10, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, tmp));
    d10 = _mm256_blendv_pd(tmp, d10, cmp);

    bitonic_sort_uint64_t_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_02v_merge_ascending(d09, d10);
}

static inline void bitonic_sort_uint64_t_10v_merge_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d09, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, tmp));
    d09 = _mm256_blendv_pd(tmp, d09, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d10, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, tmp));
    d10 = _mm256_blendv_pd(tmp, d10, cmp);

    bitonic_sort_uint64_t_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_02v_merge_descending(d09, d10);
}

static inline void bitonic_sort_uint64_t_11v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_03v_descending(d09, d10, d11);

    tmp = d09;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d09));
    d09 = _mm256_blendv_pd(d09, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d10;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d10));
    d10 = _mm256_blendv_pd(d10, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d11;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d11));
    d11 = _mm256_blendv_pd(d11, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    bitonic_sort_uint64_t_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_03v_merge_ascending(d09, d10, d11);
}

static inline void bitonic_sort_uint64_t_11v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_03v_ascending(d09, d10, d11);

    tmp = d09;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d09));
    d09 = _mm256_blendv_pd(d09, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d10;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d10));
    d10 = _mm256_blendv_pd(d10, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d11;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d11));
    d11 = _mm256_blendv_pd(d11, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    bitonic_sort_uint64_t_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_03v_merge_descending(d09, d10, d11);
}

static inline void bitonic_sort_uint64_t_11v_merge_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d09, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, tmp));
    d09 = _mm256_blendv_pd(tmp, d09, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d10, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, tmp));
    d10 = _mm256_blendv_pd(tmp, d10, cmp);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d11, d03, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, tmp));
    d11 = _mm256_blendv_pd(tmp, d11, cmp);

    bitonic_sort_uint64_t_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_03v_merge_ascending(d09, d10, d11);
}

static inline void bitonic_sort_uint64_t_11v_merge_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d09, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, tmp));
    d09 = _mm256_blendv_pd(tmp, d09, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d10, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, tmp));
    d10 = _mm256_blendv_pd(tmp, d10, cmp);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d11, d03, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, tmp));
    d11 = _mm256_blendv_pd(tmp, d11, cmp);

    bitonic_sort_uint64_t_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_03v_merge_descending(d09, d10, d11);
}

static inline void bitonic_sort_uint64_t_12v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_04v_descending(d09, d10, d11, d12);

    tmp = d09;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d09));
    d09 = _mm256_blendv_pd(d09, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d10;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d10));
    d10 = _mm256_blendv_pd(d10, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d11;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d11));
    d11 = _mm256_blendv_pd(d11, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d12;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d12));
    d12 = _mm256_blendv_pd(d12, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    bitonic_sort_uint64_t_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_04v_merge_ascending(d09, d10, d11, d12);
}

static inline void bitonic_sort_uint64_t_12v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_04v_ascending(d09, d10, d11, d12);

    tmp = d09;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d09));
    d09 = _mm256_blendv_pd(d09, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d10;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d10));
    d10 = _mm256_blendv_pd(d10, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d11;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d11));
    d11 = _mm256_blendv_pd(d11, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d12;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d12));
    d12 = _mm256_blendv_pd(d12, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    bitonic_sort_uint64_t_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_04v_merge_descending(d09, d10, d11, d12);
}

static inline void bitonic_sort_uint64_t_12v_merge_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d09, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, tmp));
    d09 = _mm256_blendv_pd(tmp, d09, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d10, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, tmp));
    d10 = _mm256_blendv_pd(tmp, d10, cmp);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d11, d03, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, tmp));
    d11 = _mm256_blendv_pd(tmp, d11, cmp);

    tmp = d04;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d04));
    d04 = _mm256_blendv_pd(d12, d04, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, tmp));
    d12 = _mm256_blendv_pd(tmp, d12, cmp);

    bitonic_sort_uint64_t_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_04v_merge_ascending(d09, d10, d11, d12);
}

static inline void bitonic_sort_uint64_t_12v_merge_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d09, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, tmp));
    d09 = _mm256_blendv_pd(tmp, d09, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d10, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, tmp));
    d10 = _mm256_blendv_pd(tmp, d10, cmp);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d11, d03, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, tmp));
    d11 = _mm256_blendv_pd(tmp, d11, cmp);

    tmp = d04;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d04));
    d04 = _mm256_blendv_pd(d12, d04, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, tmp));
    d12 = _mm256_blendv_pd(tmp, d12, cmp);

    bitonic_sort_uint64_t_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_04v_merge_descending(d09, d10, d11, d12);
}

static inline void bitonic_sort_uint64_t_13v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_05v_descending(d09, d10, d11, d12, d13);

    tmp = d09;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d09));
    d09 = _mm256_blendv_pd(d09, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d10;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d10));
    d10 = _mm256_blendv_pd(d10, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d11;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d11));
    d11 = _mm256_blendv_pd(d11, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d12;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d12));
    d12 = _mm256_blendv_pd(d12, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    tmp = d13;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d13));
    d13 = _mm256_blendv_pd(d13, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    bitonic_sort_uint64_t_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_05v_merge_ascending(d09, d10, d11, d12, d13);
}

static inline void bitonic_sort_uint64_t_13v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_05v_ascending(d09, d10, d11, d12, d13);

    tmp = d09;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d09));
    d09 = _mm256_blendv_pd(d09, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d10;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d10));
    d10 = _mm256_blendv_pd(d10, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d11;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d11));
    d11 = _mm256_blendv_pd(d11, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d12;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d12));
    d12 = _mm256_blendv_pd(d12, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    tmp = d13;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d13));
    d13 = _mm256_blendv_pd(d13, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    bitonic_sort_uint64_t_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_05v_merge_descending(d09, d10, d11, d12, d13);
}

static inline void bitonic_sort_uint64_t_13v_merge_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d09, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, tmp));
    d09 = _mm256_blendv_pd(tmp, d09, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d10, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, tmp));
    d10 = _mm256_blendv_pd(tmp, d10, cmp);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d11, d03, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, tmp));
    d11 = _mm256_blendv_pd(tmp, d11, cmp);

    tmp = d04;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d04));
    d04 = _mm256_blendv_pd(d12, d04, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, tmp));
    d12 = _mm256_blendv_pd(tmp, d12, cmp);

    tmp = d05;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d05));
    d05 = _mm256_blendv_pd(d13, d05, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, tmp));
    d13 = _mm256_blendv_pd(tmp, d13, cmp);

    bitonic_sort_uint64_t_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_05v_merge_ascending(d09, d10, d11, d12, d13);
}

static inline void bitonic_sort_uint64_t_13v_merge_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d09, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, tmp));
    d09 = _mm256_blendv_pd(tmp, d09, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d10, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, tmp));
    d10 = _mm256_blendv_pd(tmp, d10, cmp);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d11, d03, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, tmp));
    d11 = _mm256_blendv_pd(tmp, d11, cmp);

    tmp = d04;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d04));
    d04 = _mm256_blendv_pd(d12, d04, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, tmp));
    d12 = _mm256_blendv_pd(tmp, d12, cmp);

    tmp = d05;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d05));
    d05 = _mm256_blendv_pd(d13, d05, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, tmp));
    d13 = _mm256_blendv_pd(tmp, d13, cmp);

    bitonic_sort_uint64_t_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_05v_merge_descending(d09, d10, d11, d12, d13);
}

static inline void bitonic_sort_uint64_t_14v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_06v_descending(d09, d10, d11, d12, d13, d14);

    tmp = d09;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d09));
    d09 = _mm256_blendv_pd(d09, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d10;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d10));
    d10 = _mm256_blendv_pd(d10, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d11;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d11));
    d11 = _mm256_blendv_pd(d11, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d12;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d12));
    d12 = _mm256_blendv_pd(d12, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    tmp = d13;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d13));
    d13 = _mm256_blendv_pd(d13, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d14;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d14));
    d14 = _mm256_blendv_pd(d14, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    bitonic_sort_uint64_t_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_06v_merge_ascending(d09, d10, d11, d12, d13, d14);
}

static inline void bitonic_sort_uint64_t_14v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_06v_ascending(d09, d10, d11, d12, d13, d14);

    tmp = d09;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d09));
    d09 = _mm256_blendv_pd(d09, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d10;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d10));
    d10 = _mm256_blendv_pd(d10, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d11;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d11));
    d11 = _mm256_blendv_pd(d11, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d12;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d12));
    d12 = _mm256_blendv_pd(d12, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    tmp = d13;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d13));
    d13 = _mm256_blendv_pd(d13, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d14;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d14));
    d14 = _mm256_blendv_pd(d14, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    bitonic_sort_uint64_t_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_06v_merge_descending(d09, d10, d11, d12, d13, d14);
}

static inline void bitonic_sort_uint64_t_14v_merge_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d09, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, tmp));
    d09 = _mm256_blendv_pd(tmp, d09, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d10, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, tmp));
    d10 = _mm256_blendv_pd(tmp, d10, cmp);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d11, d03, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, tmp));
    d11 = _mm256_blendv_pd(tmp, d11, cmp);

    tmp = d04;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d04));
    d04 = _mm256_blendv_pd(d12, d04, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, tmp));
    d12 = _mm256_blendv_pd(tmp, d12, cmp);

    tmp = d05;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d05));
    d05 = _mm256_blendv_pd(d13, d05, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, tmp));
    d13 = _mm256_blendv_pd(tmp, d13, cmp);

    tmp = d06;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d06));
    d06 = _mm256_blendv_pd(d14, d06, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, tmp));
    d14 = _mm256_blendv_pd(tmp, d14, cmp);

    bitonic_sort_uint64_t_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_06v_merge_ascending(d09, d10, d11, d12, d13, d14);
}

static inline void bitonic_sort_uint64_t_14v_merge_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d09, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, tmp));
    d09 = _mm256_blendv_pd(tmp, d09, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d10, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, tmp));
    d10 = _mm256_blendv_pd(tmp, d10, cmp);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d11, d03, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, tmp));
    d11 = _mm256_blendv_pd(tmp, d11, cmp);

    tmp = d04;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d04));
    d04 = _mm256_blendv_pd(d12, d04, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, tmp));
    d12 = _mm256_blendv_pd(tmp, d12, cmp);

    tmp = d05;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d05));
    d05 = _mm256_blendv_pd(d13, d05, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, tmp));
    d13 = _mm256_blendv_pd(tmp, d13, cmp);

    tmp = d06;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d06));
    d06 = _mm256_blendv_pd(d14, d06, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, tmp));
    d14 = _mm256_blendv_pd(tmp, d14, cmp);

    bitonic_sort_uint64_t_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_06v_merge_descending(d09, d10, d11, d12, d13, d14);
}

static inline void bitonic_sort_uint64_t_15v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_07v_descending(d09, d10, d11, d12, d13, d14, d15);

    tmp = d09;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d09));
    d09 = _mm256_blendv_pd(d09, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d10;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d10));
    d10 = _mm256_blendv_pd(d10, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d11;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d11));
    d11 = _mm256_blendv_pd(d11, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d12;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d12));
    d12 = _mm256_blendv_pd(d12, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    tmp = d13;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d13));
    d13 = _mm256_blendv_pd(d13, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d14;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d14));
    d14 = _mm256_blendv_pd(d14, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    tmp = d15;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d15));
    d15 = _mm256_blendv_pd(d15, d02, cmp);
    d02 = _mm256_blendv_pd(d02, tmp, cmp);

    bitonic_sort_uint64_t_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_07v_merge_ascending(d09, d10, d11, d12, d13, d14, d15);
}

static inline void bitonic_sort_uint64_t_15v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_07v_ascending(d09, d10, d11, d12, d13, d14, d15);

    tmp = d09;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d09));
    d09 = _mm256_blendv_pd(d09, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d10;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d10));
    d10 = _mm256_blendv_pd(d10, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d11;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d11));
    d11 = _mm256_blendv_pd(d11, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d12;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d12));
    d12 = _mm256_blendv_pd(d12, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    tmp = d13;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d13));
    d13 = _mm256_blendv_pd(d13, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d14;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d14));
    d14 = _mm256_blendv_pd(d14, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    tmp = d15;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d15));
    d15 = _mm256_blendv_pd(d15, d02, cmp);
    d02 = _mm256_blendv_pd(d02, tmp, cmp);

    bitonic_sort_uint64_t_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_07v_merge_descending(d09, d10, d11, d12, d13, d14, d15);
}

static inline void bitonic_sort_uint64_t_15v_merge_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d09, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, tmp));
    d09 = _mm256_blendv_pd(tmp, d09, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d10, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, tmp));
    d10 = _mm256_blendv_pd(tmp, d10, cmp);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d11, d03, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, tmp));
    d11 = _mm256_blendv_pd(tmp, d11, cmp);

    tmp = d04;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d04));
    d04 = _mm256_blendv_pd(d12, d04, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, tmp));
    d12 = _mm256_blendv_pd(tmp, d12, cmp);

    tmp = d05;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d05));
    d05 = _mm256_blendv_pd(d13, d05, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, tmp));
    d13 = _mm256_blendv_pd(tmp, d13, cmp);

    tmp = d06;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d06));
    d06 = _mm256_blendv_pd(d14, d06, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, tmp));
    d14 = _mm256_blendv_pd(tmp, d14, cmp);

    tmp = d07;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d07));
    d07 = _mm256_blendv_pd(d15, d07, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, tmp));
    d15 = _mm256_blendv_pd(tmp, d15, cmp);

    bitonic_sort_uint64_t_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_07v_merge_ascending(d09, d10, d11, d12, d13, d14, d15);
}

static inline void bitonic_sort_uint64_t_15v_merge_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d09, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, tmp));
    d09 = _mm256_blendv_pd(tmp, d09, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d10, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, tmp));
    d10 = _mm256_blendv_pd(tmp, d10, cmp);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d11, d03, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, tmp));
    d11 = _mm256_blendv_pd(tmp, d11, cmp);

    tmp = d04;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d04));
    d04 = _mm256_blendv_pd(d12, d04, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, tmp));
    d12 = _mm256_blendv_pd(tmp, d12, cmp);

    tmp = d05;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d05));
    d05 = _mm256_blendv_pd(d13, d05, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, tmp));
    d13 = _mm256_blendv_pd(tmp, d13, cmp);

    tmp = d06;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d06));
    d06 = _mm256_blendv_pd(d14, d06, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, tmp));
    d14 = _mm256_blendv_pd(tmp, d14, cmp);

    tmp = d07;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d07));
    d07 = _mm256_blendv_pd(d15, d07, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, tmp));
    d15 = _mm256_blendv_pd(tmp, d15, cmp);

    bitonic_sort_uint64_t_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_07v_merge_descending(d09, d10, d11, d12, d13, d14, d15);
}

static inline void bitonic_sort_uint64_t_16v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_08v_descending(d09, d10, d11, d12, d13, d14, d15, d16);

    tmp = d09;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d09));
    d09 = _mm256_blendv_pd(d09, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d10;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d10));
    d10 = _mm256_blendv_pd(d10, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d11;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d11));
    d11 = _mm256_blendv_pd(d11, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d12;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d12));
    d12 = _mm256_blendv_pd(d12, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    tmp = d13;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d13));
    d13 = _mm256_blendv_pd(d13, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d14;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d14));
    d14 = _mm256_blendv_pd(d14, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    tmp = d15;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d15));
    d15 = _mm256_blendv_pd(d15, d02, cmp);
    d02 = _mm256_blendv_pd(d02, tmp, cmp);

    tmp = d16;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d01), _mm256_xor_si256(topBit, d16));
    d16 = _mm256_blendv_pd(d16, d01, cmp);
    d01 = _mm256_blendv_pd(d01, tmp, cmp);

    bitonic_sort_uint64_t_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_08v_merge_ascending(d09, d10, d11, d12, d13, d14, d15, d16);
}

static inline void bitonic_sort_uint64_t_16v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_08v_ascending(d09, d10, d11, d12, d13, d14, d15, d16);

    tmp = d09;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d09));
    d09 = _mm256_blendv_pd(d09, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d10;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d10));
    d10 = _mm256_blendv_pd(d10, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d11;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d11));
    d11 = _mm256_blendv_pd(d11, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d12;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d12));
    d12 = _mm256_blendv_pd(d12, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    tmp = d13;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d13));
    d13 = _mm256_blendv_pd(d13, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d14;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d14));
    d14 = _mm256_blendv_pd(d14, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    tmp = d15;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d15));
    d15 = _mm256_blendv_pd(d15, d02, cmp);
    d02 = _mm256_blendv_pd(d02, tmp, cmp);

    tmp = d16;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d01), _mm256_xor_si256(topBit, d16));
    d16 = _mm256_blendv_pd(d16, d01, cmp);
    d01 = _mm256_blendv_pd(d01, tmp, cmp);

    bitonic_sort_uint64_t_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_08v_merge_descending(d09, d10, d11, d12, d13, d14, d15, d16);
}

static inline void bitonic_sort_uint64_t_16v_merge_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d09, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, tmp));
    d09 = _mm256_blendv_pd(tmp, d09, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d10, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, tmp));
    d10 = _mm256_blendv_pd(tmp, d10, cmp);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d11, d03, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, tmp));
    d11 = _mm256_blendv_pd(tmp, d11, cmp);

    tmp = d04;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d04));
    d04 = _mm256_blendv_pd(d12, d04, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, tmp));
    d12 = _mm256_blendv_pd(tmp, d12, cmp);

    tmp = d05;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d05));
    d05 = _mm256_blendv_pd(d13, d05, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, tmp));
    d13 = _mm256_blendv_pd(tmp, d13, cmp);

    tmp = d06;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d06));
    d06 = _mm256_blendv_pd(d14, d06, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, tmp));
    d14 = _mm256_blendv_pd(tmp, d14, cmp);

    tmp = d07;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d07));
    d07 = _mm256_blendv_pd(d15, d07, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, tmp));
    d15 = _mm256_blendv_pd(tmp, d15, cmp);

    tmp = d08;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d08));
    d08 = _mm256_blendv_pd(d16, d08, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, tmp));
    d16 = _mm256_blendv_pd(tmp, d16, cmp);

    bitonic_sort_uint64_t_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_08v_merge_ascending(d09, d10, d11, d12, d13, d14, d15, d16);
}

static inline void bitonic_sort_uint64_t_16v_merge_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d01));
    d01 = _mm256_blendv_pd(d09, d01, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, tmp));
    d09 = _mm256_blendv_pd(tmp, d09, cmp);

    tmp = d02;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d02));
    d02 = _mm256_blendv_pd(d10, d02, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, tmp));
    d10 = _mm256_blendv_pd(tmp, d10, cmp);

    tmp = d03;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d03));
    d03 = _mm256_blendv_pd(d11, d03, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, tmp));
    d11 = _mm256_blendv_pd(tmp, d11, cmp);

    tmp = d04;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d04));
    d04 = _mm256_blendv_pd(d12, d04, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, tmp));
    d12 = _mm256_blendv_pd(tmp, d12, cmp);

    tmp = d05;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d05));
    d05 = _mm256_blendv_pd(d13, d05, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, tmp));
    d13 = _mm256_blendv_pd(tmp, d13, cmp);

    tmp = d06;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d06));
    d06 = _mm256_blendv_pd(d14, d06, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, tmp));
    d14 = _mm256_blendv_pd(tmp, d14, cmp);

    tmp = d07;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d07));
    d07 = _mm256_blendv_pd(d15, d07, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, tmp));
    d15 = _mm256_blendv_pd(tmp, d15, cmp);

    tmp = d08;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d08));
    d08 = _mm256_blendv_pd(d16, d08, cmp);
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, tmp));
    d16 = _mm256_blendv_pd(tmp, d16, cmp);

    bitonic_sort_uint64_t_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_uint64_t_08v_merge_descending(d09, d10, d11, d12, d13, d14, d15, d16);
}

static inline void bitonic_sort_uint64_t_17v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_01v_descending(d17);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_01v_merge_ascending(d17);
}

static inline void bitonic_sort_uint64_t_17v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_01v_ascending(d17);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_01v_merge_descending(d17);
}

static inline void bitonic_sort_uint64_t_18v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_02v_descending(d17, d18);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_02v_merge_ascending(d17, d18);
}

static inline void bitonic_sort_uint64_t_18v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_02v_ascending(d17, d18);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_02v_merge_descending(d17, d18);
}

static inline void bitonic_sort_uint64_t_19v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_03v_descending(d17, d18, d19);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_03v_merge_ascending(d17, d18, d19);
}

static inline void bitonic_sort_uint64_t_19v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_03v_ascending(d17, d18, d19);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_03v_merge_descending(d17, d18, d19);
}

static inline void bitonic_sort_uint64_t_20v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_04v_descending(d17, d18, d19, d20);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_04v_merge_ascending(d17, d18, d19, d20);
}

static inline void bitonic_sort_uint64_t_20v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_04v_ascending(d17, d18, d19, d20);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_04v_merge_descending(d17, d18, d19, d20);
}

static inline void bitonic_sort_uint64_t_21v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_05v_descending(d17, d18, d19, d20, d21);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_05v_merge_ascending(d17, d18, d19, d20, d21);
}

static inline void bitonic_sort_uint64_t_21v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_05v_ascending(d17, d18, d19, d20, d21);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_05v_merge_descending(d17, d18, d19, d20, d21);
}

static inline void bitonic_sort_uint64_t_22v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_06v_descending(d17, d18, d19, d20, d21, d22);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_06v_merge_ascending(d17, d18, d19, d20, d21, d22);
}

static inline void bitonic_sort_uint64_t_22v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_06v_ascending(d17, d18, d19, d20, d21, d22);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_06v_merge_descending(d17, d18, d19, d20, d21, d22);
}

static inline void bitonic_sort_uint64_t_23v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_07v_descending(d17, d18, d19, d20, d21, d22, d23);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_07v_merge_ascending(d17, d18, d19, d20, d21, d22, d23);
}

static inline void bitonic_sort_uint64_t_23v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_07v_ascending(d17, d18, d19, d20, d21, d22, d23);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_07v_merge_descending(d17, d18, d19, d20, d21, d22, d23);
}

static inline void bitonic_sort_uint64_t_24v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_08v_descending(d17, d18, d19, d20, d21, d22, d23, d24);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_08v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24);
}

static inline void bitonic_sort_uint64_t_24v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_08v_ascending(d17, d18, d19, d20, d21, d22, d23, d24);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_08v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24);
}

static inline void bitonic_sort_uint64_t_25v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24, __m256i& d25) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_09v_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    tmp = d25;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d25));
    d25 = _mm256_blendv_pd(d25, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_09v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25);
}

static inline void bitonic_sort_uint64_t_25v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24, __m256i& d25) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_09v_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    tmp = d25;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d25));
    d25 = _mm256_blendv_pd(d25, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_09v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25);
}

static inline void bitonic_sort_uint64_t_26v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24, __m256i& d25, __m256i& d26) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_10v_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    tmp = d25;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d25));
    d25 = _mm256_blendv_pd(d25, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d26;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d26));
    d26 = _mm256_blendv_pd(d26, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_10v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26);
}

static inline void bitonic_sort_uint64_t_26v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24, __m256i& d25, __m256i& d26) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_10v_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    tmp = d25;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d25));
    d25 = _mm256_blendv_pd(d25, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d26;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d26));
    d26 = _mm256_blendv_pd(d26, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_10v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26);
}

static inline void bitonic_sort_uint64_t_27v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24, __m256i& d25, __m256i& d26, __m256i& d27) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_11v_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    tmp = d25;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d25));
    d25 = _mm256_blendv_pd(d25, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d26;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d26));
    d26 = _mm256_blendv_pd(d26, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d27;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d27));
    d27 = _mm256_blendv_pd(d27, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_11v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27);
}

static inline void bitonic_sort_uint64_t_27v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24, __m256i& d25, __m256i& d26, __m256i& d27) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_11v_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    tmp = d25;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d25));
    d25 = _mm256_blendv_pd(d25, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d26;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d26));
    d26 = _mm256_blendv_pd(d26, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d27;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d27));
    d27 = _mm256_blendv_pd(d27, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_11v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27);
}

static inline void bitonic_sort_uint64_t_28v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24, __m256i& d25, __m256i& d26, __m256i& d27, __m256i& d28) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_12v_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    tmp = d25;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d25));
    d25 = _mm256_blendv_pd(d25, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d26;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d26));
    d26 = _mm256_blendv_pd(d26, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d27;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d27));
    d27 = _mm256_blendv_pd(d27, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d28;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d28));
    d28 = _mm256_blendv_pd(d28, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_12v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28);
}

static inline void bitonic_sort_uint64_t_28v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24, __m256i& d25, __m256i& d26, __m256i& d27, __m256i& d28) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_12v_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    tmp = d25;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d25));
    d25 = _mm256_blendv_pd(d25, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d26;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d26));
    d26 = _mm256_blendv_pd(d26, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d27;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d27));
    d27 = _mm256_blendv_pd(d27, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d28;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d28));
    d28 = _mm256_blendv_pd(d28, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_12v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28);
}

static inline void bitonic_sort_uint64_t_29v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24, __m256i& d25, __m256i& d26, __m256i& d27, __m256i& d28, __m256i& d29) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_13v_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    tmp = d25;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d25));
    d25 = _mm256_blendv_pd(d25, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d26;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d26));
    d26 = _mm256_blendv_pd(d26, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d27;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d27));
    d27 = _mm256_blendv_pd(d27, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d28;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d28));
    d28 = _mm256_blendv_pd(d28, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    tmp = d29;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d29));
    d29 = _mm256_blendv_pd(d29, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_13v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29);
}

static inline void bitonic_sort_uint64_t_29v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24, __m256i& d25, __m256i& d26, __m256i& d27, __m256i& d28, __m256i& d29) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_13v_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    tmp = d25;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d25));
    d25 = _mm256_blendv_pd(d25, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d26;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d26));
    d26 = _mm256_blendv_pd(d26, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d27;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d27));
    d27 = _mm256_blendv_pd(d27, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d28;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d28));
    d28 = _mm256_blendv_pd(d28, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    tmp = d29;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d29));
    d29 = _mm256_blendv_pd(d29, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_13v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29);
}

static inline void bitonic_sort_uint64_t_30v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24, __m256i& d25, __m256i& d26, __m256i& d27, __m256i& d28, __m256i& d29, __m256i& d30) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_14v_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    tmp = d25;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d25));
    d25 = _mm256_blendv_pd(d25, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d26;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d26));
    d26 = _mm256_blendv_pd(d26, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d27;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d27));
    d27 = _mm256_blendv_pd(d27, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d28;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d28));
    d28 = _mm256_blendv_pd(d28, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    tmp = d29;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d29));
    d29 = _mm256_blendv_pd(d29, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d30;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d30));
    d30 = _mm256_blendv_pd(d30, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_14v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30);
}

static inline void bitonic_sort_uint64_t_30v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24, __m256i& d25, __m256i& d26, __m256i& d27, __m256i& d28, __m256i& d29, __m256i& d30) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_14v_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    tmp = d25;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d25));
    d25 = _mm256_blendv_pd(d25, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d26;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d26));
    d26 = _mm256_blendv_pd(d26, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d27;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d27));
    d27 = _mm256_blendv_pd(d27, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d28;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d28));
    d28 = _mm256_blendv_pd(d28, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    tmp = d29;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d29));
    d29 = _mm256_blendv_pd(d29, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d30;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d30));
    d30 = _mm256_blendv_pd(d30, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_14v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30);
}

static inline void bitonic_sort_uint64_t_31v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24, __m256i& d25, __m256i& d26, __m256i& d27, __m256i& d28, __m256i& d29, __m256i& d30, __m256i& d31) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_15v_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    tmp = d25;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d25));
    d25 = _mm256_blendv_pd(d25, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d26;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d26));
    d26 = _mm256_blendv_pd(d26, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d27;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d27));
    d27 = _mm256_blendv_pd(d27, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d28;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d28));
    d28 = _mm256_blendv_pd(d28, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    tmp = d29;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d29));
    d29 = _mm256_blendv_pd(d29, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d30;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d30));
    d30 = _mm256_blendv_pd(d30, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    tmp = d31;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d31));
    d31 = _mm256_blendv_pd(d31, d02, cmp);
    d02 = _mm256_blendv_pd(d02, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_15v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31);
}

static inline void bitonic_sort_uint64_t_31v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24, __m256i& d25, __m256i& d26, __m256i& d27, __m256i& d28, __m256i& d29, __m256i& d30, __m256i& d31) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_15v_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    tmp = d25;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d25));
    d25 = _mm256_blendv_pd(d25, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d26;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d26));
    d26 = _mm256_blendv_pd(d26, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d27;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d27));
    d27 = _mm256_blendv_pd(d27, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d28;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d28));
    d28 = _mm256_blendv_pd(d28, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    tmp = d29;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d29));
    d29 = _mm256_blendv_pd(d29, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d30;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d30));
    d30 = _mm256_blendv_pd(d30, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    tmp = d31;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d31));
    d31 = _mm256_blendv_pd(d31, d02, cmp);
    d02 = _mm256_blendv_pd(d02, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_15v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31);
}

static inline void bitonic_sort_uint64_t_32v_ascending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24, __m256i& d25, __m256i& d26, __m256i& d27, __m256i& d28, __m256i& d29, __m256i& d30, __m256i& d31, __m256i& d32) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_16v_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    tmp = d25;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d25));
    d25 = _mm256_blendv_pd(d25, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d26;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d26));
    d26 = _mm256_blendv_pd(d26, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d27;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d27));
    d27 = _mm256_blendv_pd(d27, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d28;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d28));
    d28 = _mm256_blendv_pd(d28, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    tmp = d29;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d29));
    d29 = _mm256_blendv_pd(d29, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d30;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d30));
    d30 = _mm256_blendv_pd(d30, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    tmp = d31;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d31));
    d31 = _mm256_blendv_pd(d31, d02, cmp);
    d02 = _mm256_blendv_pd(d02, tmp, cmp);

    tmp = d32;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d01), _mm256_xor_si256(topBit, d32));
    d32 = _mm256_blendv_pd(d32, d01, cmp);
    d01 = _mm256_blendv_pd(d01, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_16v_merge_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32);
}

static inline void bitonic_sort_uint64_t_32v_descending(__m256i& d01, __m256i& d02, __m256i& d03, __m256i& d04, __m256i& d05, __m256i& d06, __m256i& d07, __m256i& d08, __m256i& d09, __m256i& d10, __m256i& d11, __m256i& d12, __m256i& d13, __m256i& d14, __m256i& d15, __m256i& d16, __m256i& d17, __m256i& d18, __m256i& d19, __m256i& d20, __m256i& d21, __m256i& d22, __m256i& d23, __m256i& d24, __m256i& d25, __m256i& d26, __m256i& d27, __m256i& d28, __m256i& d29, __m256i& d30, __m256i& d31, __m256i& d32) {
    __m256i  tmp, cmp;
    __m256i topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_uint64_t_16v_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_16v_ascending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32);

    tmp = d17;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d16), _mm256_xor_si256(topBit, d17));
    d17 = _mm256_blendv_pd(d17, d16, cmp);
    d16 = _mm256_blendv_pd(d16, tmp, cmp);

    tmp = d18;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d15), _mm256_xor_si256(topBit, d18));
    d18 = _mm256_blendv_pd(d18, d15, cmp);
    d15 = _mm256_blendv_pd(d15, tmp, cmp);

    tmp = d19;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d14), _mm256_xor_si256(topBit, d19));
    d19 = _mm256_blendv_pd(d19, d14, cmp);
    d14 = _mm256_blendv_pd(d14, tmp, cmp);

    tmp = d20;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d13), _mm256_xor_si256(topBit, d20));
    d20 = _mm256_blendv_pd(d20, d13, cmp);
    d13 = _mm256_blendv_pd(d13, tmp, cmp);

    tmp = d21;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d12), _mm256_xor_si256(topBit, d21));
    d21 = _mm256_blendv_pd(d21, d12, cmp);
    d12 = _mm256_blendv_pd(d12, tmp, cmp);

    tmp = d22;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d11), _mm256_xor_si256(topBit, d22));
    d22 = _mm256_blendv_pd(d22, d11, cmp);
    d11 = _mm256_blendv_pd(d11, tmp, cmp);

    tmp = d23;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d10), _mm256_xor_si256(topBit, d23));
    d23 = _mm256_blendv_pd(d23, d10, cmp);
    d10 = _mm256_blendv_pd(d10, tmp, cmp);

    tmp = d24;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d09), _mm256_xor_si256(topBit, d24));
    d24 = _mm256_blendv_pd(d24, d09, cmp);
    d09 = _mm256_blendv_pd(d09, tmp, cmp);

    tmp = d25;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d08), _mm256_xor_si256(topBit, d25));
    d25 = _mm256_blendv_pd(d25, d08, cmp);
    d08 = _mm256_blendv_pd(d08, tmp, cmp);

    tmp = d26;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d07), _mm256_xor_si256(topBit, d26));
    d26 = _mm256_blendv_pd(d26, d07, cmp);
    d07 = _mm256_blendv_pd(d07, tmp, cmp);

    tmp = d27;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d06), _mm256_xor_si256(topBit, d27));
    d27 = _mm256_blendv_pd(d27, d06, cmp);
    d06 = _mm256_blendv_pd(d06, tmp, cmp);

    tmp = d28;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d05), _mm256_xor_si256(topBit, d28));
    d28 = _mm256_blendv_pd(d28, d05, cmp);
    d05 = _mm256_blendv_pd(d05, tmp, cmp);

    tmp = d29;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d04), _mm256_xor_si256(topBit, d29));
    d29 = _mm256_blendv_pd(d29, d04, cmp);
    d04 = _mm256_blendv_pd(d04, tmp, cmp);

    tmp = d30;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d03), _mm256_xor_si256(topBit, d30));
    d30 = _mm256_blendv_pd(d30, d03, cmp);
    d03 = _mm256_blendv_pd(d03, tmp, cmp);

    tmp = d31;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d02), _mm256_xor_si256(topBit, d31));
    d31 = _mm256_blendv_pd(d31, d02, cmp);
    d02 = _mm256_blendv_pd(d02, tmp, cmp);

    tmp = d32;
    cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, d01), _mm256_xor_si256(topBit, d32));
    d32 = _mm256_blendv_pd(d32, d01, cmp);
    d01 = _mm256_blendv_pd(d01, tmp, cmp);

    bitonic_sort_uint64_t_16v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    bitonic_sort_uint64_t_16v_merge_descending(d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_01v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    bitonic_sort_uint64_t_01v_ascending(d01);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_02v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    bitonic_sort_uint64_t_02v_ascending(d01, d02);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_03v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    bitonic_sort_uint64_t_03v_ascending(d01, d02, d03);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_04v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    bitonic_sort_uint64_t_04v_ascending(d01, d02, d03, d04);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_05v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    bitonic_sort_uint64_t_05v_ascending(d01, d02, d03, d04, d05);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_06v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    bitonic_sort_uint64_t_06v_ascending(d01, d02, d03, d04, d05, d06);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_07v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    bitonic_sort_uint64_t_07v_ascending(d01, d02, d03, d04, d05, d06, d07);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_08v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    bitonic_sort_uint64_t_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_09v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    bitonic_sort_uint64_t_09v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_10v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    bitonic_sort_uint64_t_10v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_11v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    bitonic_sort_uint64_t_11v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_12v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    bitonic_sort_uint64_t_12v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_13v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    bitonic_sort_uint64_t_13v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_14v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    bitonic_sort_uint64_t_14v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_15v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    bitonic_sort_uint64_t_15v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_16v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    bitonic_sort_uint64_t_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_17v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    __m256i d17 = _mm256_lddqu_si256((__m256i const *) ptr + 16);
    bitonic_sort_uint64_t_17v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 16, d17);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_18v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    __m256i d17 = _mm256_lddqu_si256((__m256i const *) ptr + 16);
    __m256i d18 = _mm256_lddqu_si256((__m256i const *) ptr + 17);
    bitonic_sort_uint64_t_18v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 16, d17);
    _mm256_storeu_si256((__m256i *) ptr + 17, d18);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_19v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    __m256i d17 = _mm256_lddqu_si256((__m256i const *) ptr + 16);
    __m256i d18 = _mm256_lddqu_si256((__m256i const *) ptr + 17);
    __m256i d19 = _mm256_lddqu_si256((__m256i const *) ptr + 18);
    bitonic_sort_uint64_t_19v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 16, d17);
    _mm256_storeu_si256((__m256i *) ptr + 17, d18);
    _mm256_storeu_si256((__m256i *) ptr + 18, d19);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_20v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    __m256i d17 = _mm256_lddqu_si256((__m256i const *) ptr + 16);
    __m256i d18 = _mm256_lddqu_si256((__m256i const *) ptr + 17);
    __m256i d19 = _mm256_lddqu_si256((__m256i const *) ptr + 18);
    __m256i d20 = _mm256_lddqu_si256((__m256i const *) ptr + 19);
    bitonic_sort_uint64_t_20v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 16, d17);
    _mm256_storeu_si256((__m256i *) ptr + 17, d18);
    _mm256_storeu_si256((__m256i *) ptr + 18, d19);
    _mm256_storeu_si256((__m256i *) ptr + 19, d20);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_21v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    __m256i d17 = _mm256_lddqu_si256((__m256i const *) ptr + 16);
    __m256i d18 = _mm256_lddqu_si256((__m256i const *) ptr + 17);
    __m256i d19 = _mm256_lddqu_si256((__m256i const *) ptr + 18);
    __m256i d20 = _mm256_lddqu_si256((__m256i const *) ptr + 19);
    __m256i d21 = _mm256_lddqu_si256((__m256i const *) ptr + 20);
    bitonic_sort_uint64_t_21v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 16, d17);
    _mm256_storeu_si256((__m256i *) ptr + 17, d18);
    _mm256_storeu_si256((__m256i *) ptr + 18, d19);
    _mm256_storeu_si256((__m256i *) ptr + 19, d20);
    _mm256_storeu_si256((__m256i *) ptr + 20, d21);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_22v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    __m256i d17 = _mm256_lddqu_si256((__m256i const *) ptr + 16);
    __m256i d18 = _mm256_lddqu_si256((__m256i const *) ptr + 17);
    __m256i d19 = _mm256_lddqu_si256((__m256i const *) ptr + 18);
    __m256i d20 = _mm256_lddqu_si256((__m256i const *) ptr + 19);
    __m256i d21 = _mm256_lddqu_si256((__m256i const *) ptr + 20);
    __m256i d22 = _mm256_lddqu_si256((__m256i const *) ptr + 21);
    bitonic_sort_uint64_t_22v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 16, d17);
    _mm256_storeu_si256((__m256i *) ptr + 17, d18);
    _mm256_storeu_si256((__m256i *) ptr + 18, d19);
    _mm256_storeu_si256((__m256i *) ptr + 19, d20);
    _mm256_storeu_si256((__m256i *) ptr + 20, d21);
    _mm256_storeu_si256((__m256i *) ptr + 21, d22);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_23v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    __m256i d17 = _mm256_lddqu_si256((__m256i const *) ptr + 16);
    __m256i d18 = _mm256_lddqu_si256((__m256i const *) ptr + 17);
    __m256i d19 = _mm256_lddqu_si256((__m256i const *) ptr + 18);
    __m256i d20 = _mm256_lddqu_si256((__m256i const *) ptr + 19);
    __m256i d21 = _mm256_lddqu_si256((__m256i const *) ptr + 20);
    __m256i d22 = _mm256_lddqu_si256((__m256i const *) ptr + 21);
    __m256i d23 = _mm256_lddqu_si256((__m256i const *) ptr + 22);
    bitonic_sort_uint64_t_23v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 16, d17);
    _mm256_storeu_si256((__m256i *) ptr + 17, d18);
    _mm256_storeu_si256((__m256i *) ptr + 18, d19);
    _mm256_storeu_si256((__m256i *) ptr + 19, d20);
    _mm256_storeu_si256((__m256i *) ptr + 20, d21);
    _mm256_storeu_si256((__m256i *) ptr + 21, d22);
    _mm256_storeu_si256((__m256i *) ptr + 22, d23);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_24v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    __m256i d17 = _mm256_lddqu_si256((__m256i const *) ptr + 16);
    __m256i d18 = _mm256_lddqu_si256((__m256i const *) ptr + 17);
    __m256i d19 = _mm256_lddqu_si256((__m256i const *) ptr + 18);
    __m256i d20 = _mm256_lddqu_si256((__m256i const *) ptr + 19);
    __m256i d21 = _mm256_lddqu_si256((__m256i const *) ptr + 20);
    __m256i d22 = _mm256_lddqu_si256((__m256i const *) ptr + 21);
    __m256i d23 = _mm256_lddqu_si256((__m256i const *) ptr + 22);
    __m256i d24 = _mm256_lddqu_si256((__m256i const *) ptr + 23);
    bitonic_sort_uint64_t_24v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 16, d17);
    _mm256_storeu_si256((__m256i *) ptr + 17, d18);
    _mm256_storeu_si256((__m256i *) ptr + 18, d19);
    _mm256_storeu_si256((__m256i *) ptr + 19, d20);
    _mm256_storeu_si256((__m256i *) ptr + 20, d21);
    _mm256_storeu_si256((__m256i *) ptr + 21, d22);
    _mm256_storeu_si256((__m256i *) ptr + 22, d23);
    _mm256_storeu_si256((__m256i *) ptr + 23, d24);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_25v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    __m256i d17 = _mm256_lddqu_si256((__m256i const *) ptr + 16);
    __m256i d18 = _mm256_lddqu_si256((__m256i const *) ptr + 17);
    __m256i d19 = _mm256_lddqu_si256((__m256i const *) ptr + 18);
    __m256i d20 = _mm256_lddqu_si256((__m256i const *) ptr + 19);
    __m256i d21 = _mm256_lddqu_si256((__m256i const *) ptr + 20);
    __m256i d22 = _mm256_lddqu_si256((__m256i const *) ptr + 21);
    __m256i d23 = _mm256_lddqu_si256((__m256i const *) ptr + 22);
    __m256i d24 = _mm256_lddqu_si256((__m256i const *) ptr + 23);
    __m256i d25 = _mm256_lddqu_si256((__m256i const *) ptr + 24);
    bitonic_sort_uint64_t_25v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 16, d17);
    _mm256_storeu_si256((__m256i *) ptr + 17, d18);
    _mm256_storeu_si256((__m256i *) ptr + 18, d19);
    _mm256_storeu_si256((__m256i *) ptr + 19, d20);
    _mm256_storeu_si256((__m256i *) ptr + 20, d21);
    _mm256_storeu_si256((__m256i *) ptr + 21, d22);
    _mm256_storeu_si256((__m256i *) ptr + 22, d23);
    _mm256_storeu_si256((__m256i *) ptr + 23, d24);
    _mm256_storeu_si256((__m256i *) ptr + 24, d25);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_26v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    __m256i d17 = _mm256_lddqu_si256((__m256i const *) ptr + 16);
    __m256i d18 = _mm256_lddqu_si256((__m256i const *) ptr + 17);
    __m256i d19 = _mm256_lddqu_si256((__m256i const *) ptr + 18);
    __m256i d20 = _mm256_lddqu_si256((__m256i const *) ptr + 19);
    __m256i d21 = _mm256_lddqu_si256((__m256i const *) ptr + 20);
    __m256i d22 = _mm256_lddqu_si256((__m256i const *) ptr + 21);
    __m256i d23 = _mm256_lddqu_si256((__m256i const *) ptr + 22);
    __m256i d24 = _mm256_lddqu_si256((__m256i const *) ptr + 23);
    __m256i d25 = _mm256_lddqu_si256((__m256i const *) ptr + 24);
    __m256i d26 = _mm256_lddqu_si256((__m256i const *) ptr + 25);
    bitonic_sort_uint64_t_26v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 16, d17);
    _mm256_storeu_si256((__m256i *) ptr + 17, d18);
    _mm256_storeu_si256((__m256i *) ptr + 18, d19);
    _mm256_storeu_si256((__m256i *) ptr + 19, d20);
    _mm256_storeu_si256((__m256i *) ptr + 20, d21);
    _mm256_storeu_si256((__m256i *) ptr + 21, d22);
    _mm256_storeu_si256((__m256i *) ptr + 22, d23);
    _mm256_storeu_si256((__m256i *) ptr + 23, d24);
    _mm256_storeu_si256((__m256i *) ptr + 24, d25);
    _mm256_storeu_si256((__m256i *) ptr + 25, d26);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_27v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    __m256i d17 = _mm256_lddqu_si256((__m256i const *) ptr + 16);
    __m256i d18 = _mm256_lddqu_si256((__m256i const *) ptr + 17);
    __m256i d19 = _mm256_lddqu_si256((__m256i const *) ptr + 18);
    __m256i d20 = _mm256_lddqu_si256((__m256i const *) ptr + 19);
    __m256i d21 = _mm256_lddqu_si256((__m256i const *) ptr + 20);
    __m256i d22 = _mm256_lddqu_si256((__m256i const *) ptr + 21);
    __m256i d23 = _mm256_lddqu_si256((__m256i const *) ptr + 22);
    __m256i d24 = _mm256_lddqu_si256((__m256i const *) ptr + 23);
    __m256i d25 = _mm256_lddqu_si256((__m256i const *) ptr + 24);
    __m256i d26 = _mm256_lddqu_si256((__m256i const *) ptr + 25);
    __m256i d27 = _mm256_lddqu_si256((__m256i const *) ptr + 26);
    bitonic_sort_uint64_t_27v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 16, d17);
    _mm256_storeu_si256((__m256i *) ptr + 17, d18);
    _mm256_storeu_si256((__m256i *) ptr + 18, d19);
    _mm256_storeu_si256((__m256i *) ptr + 19, d20);
    _mm256_storeu_si256((__m256i *) ptr + 20, d21);
    _mm256_storeu_si256((__m256i *) ptr + 21, d22);
    _mm256_storeu_si256((__m256i *) ptr + 22, d23);
    _mm256_storeu_si256((__m256i *) ptr + 23, d24);
    _mm256_storeu_si256((__m256i *) ptr + 24, d25);
    _mm256_storeu_si256((__m256i *) ptr + 25, d26);
    _mm256_storeu_si256((__m256i *) ptr + 26, d27);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_28v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    __m256i d17 = _mm256_lddqu_si256((__m256i const *) ptr + 16);
    __m256i d18 = _mm256_lddqu_si256((__m256i const *) ptr + 17);
    __m256i d19 = _mm256_lddqu_si256((__m256i const *) ptr + 18);
    __m256i d20 = _mm256_lddqu_si256((__m256i const *) ptr + 19);
    __m256i d21 = _mm256_lddqu_si256((__m256i const *) ptr + 20);
    __m256i d22 = _mm256_lddqu_si256((__m256i const *) ptr + 21);
    __m256i d23 = _mm256_lddqu_si256((__m256i const *) ptr + 22);
    __m256i d24 = _mm256_lddqu_si256((__m256i const *) ptr + 23);
    __m256i d25 = _mm256_lddqu_si256((__m256i const *) ptr + 24);
    __m256i d26 = _mm256_lddqu_si256((__m256i const *) ptr + 25);
    __m256i d27 = _mm256_lddqu_si256((__m256i const *) ptr + 26);
    __m256i d28 = _mm256_lddqu_si256((__m256i const *) ptr + 27);
    bitonic_sort_uint64_t_28v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 16, d17);
    _mm256_storeu_si256((__m256i *) ptr + 17, d18);
    _mm256_storeu_si256((__m256i *) ptr + 18, d19);
    _mm256_storeu_si256((__m256i *) ptr + 19, d20);
    _mm256_storeu_si256((__m256i *) ptr + 20, d21);
    _mm256_storeu_si256((__m256i *) ptr + 21, d22);
    _mm256_storeu_si256((__m256i *) ptr + 22, d23);
    _mm256_storeu_si256((__m256i *) ptr + 23, d24);
    _mm256_storeu_si256((__m256i *) ptr + 24, d25);
    _mm256_storeu_si256((__m256i *) ptr + 25, d26);
    _mm256_storeu_si256((__m256i *) ptr + 26, d27);
    _mm256_storeu_si256((__m256i *) ptr + 27, d28);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_29v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    __m256i d17 = _mm256_lddqu_si256((__m256i const *) ptr + 16);
    __m256i d18 = _mm256_lddqu_si256((__m256i const *) ptr + 17);
    __m256i d19 = _mm256_lddqu_si256((__m256i const *) ptr + 18);
    __m256i d20 = _mm256_lddqu_si256((__m256i const *) ptr + 19);
    __m256i d21 = _mm256_lddqu_si256((__m256i const *) ptr + 20);
    __m256i d22 = _mm256_lddqu_si256((__m256i const *) ptr + 21);
    __m256i d23 = _mm256_lddqu_si256((__m256i const *) ptr + 22);
    __m256i d24 = _mm256_lddqu_si256((__m256i const *) ptr + 23);
    __m256i d25 = _mm256_lddqu_si256((__m256i const *) ptr + 24);
    __m256i d26 = _mm256_lddqu_si256((__m256i const *) ptr + 25);
    __m256i d27 = _mm256_lddqu_si256((__m256i const *) ptr + 26);
    __m256i d28 = _mm256_lddqu_si256((__m256i const *) ptr + 27);
    __m256i d29 = _mm256_lddqu_si256((__m256i const *) ptr + 28);
    bitonic_sort_uint64_t_29v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 16, d17);
    _mm256_storeu_si256((__m256i *) ptr + 17, d18);
    _mm256_storeu_si256((__m256i *) ptr + 18, d19);
    _mm256_storeu_si256((__m256i *) ptr + 19, d20);
    _mm256_storeu_si256((__m256i *) ptr + 20, d21);
    _mm256_storeu_si256((__m256i *) ptr + 21, d22);
    _mm256_storeu_si256((__m256i *) ptr + 22, d23);
    _mm256_storeu_si256((__m256i *) ptr + 23, d24);
    _mm256_storeu_si256((__m256i *) ptr + 24, d25);
    _mm256_storeu_si256((__m256i *) ptr + 25, d26);
    _mm256_storeu_si256((__m256i *) ptr + 26, d27);
    _mm256_storeu_si256((__m256i *) ptr + 27, d28);
    _mm256_storeu_si256((__m256i *) ptr + 28, d29);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_30v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    __m256i d17 = _mm256_lddqu_si256((__m256i const *) ptr + 16);
    __m256i d18 = _mm256_lddqu_si256((__m256i const *) ptr + 17);
    __m256i d19 = _mm256_lddqu_si256((__m256i const *) ptr + 18);
    __m256i d20 = _mm256_lddqu_si256((__m256i const *) ptr + 19);
    __m256i d21 = _mm256_lddqu_si256((__m256i const *) ptr + 20);
    __m256i d22 = _mm256_lddqu_si256((__m256i const *) ptr + 21);
    __m256i d23 = _mm256_lddqu_si256((__m256i const *) ptr + 22);
    __m256i d24 = _mm256_lddqu_si256((__m256i const *) ptr + 23);
    __m256i d25 = _mm256_lddqu_si256((__m256i const *) ptr + 24);
    __m256i d26 = _mm256_lddqu_si256((__m256i const *) ptr + 25);
    __m256i d27 = _mm256_lddqu_si256((__m256i const *) ptr + 26);
    __m256i d28 = _mm256_lddqu_si256((__m256i const *) ptr + 27);
    __m256i d29 = _mm256_lddqu_si256((__m256i const *) ptr + 28);
    __m256i d30 = _mm256_lddqu_si256((__m256i const *) ptr + 29);
    bitonic_sort_uint64_t_30v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 16, d17);
    _mm256_storeu_si256((__m256i *) ptr + 17, d18);
    _mm256_storeu_si256((__m256i *) ptr + 18, d19);
    _mm256_storeu_si256((__m256i *) ptr + 19, d20);
    _mm256_storeu_si256((__m256i *) ptr + 20, d21);
    _mm256_storeu_si256((__m256i *) ptr + 21, d22);
    _mm256_storeu_si256((__m256i *) ptr + 22, d23);
    _mm256_storeu_si256((__m256i *) ptr + 23, d24);
    _mm256_storeu_si256((__m256i *) ptr + 24, d25);
    _mm256_storeu_si256((__m256i *) ptr + 25, d26);
    _mm256_storeu_si256((__m256i *) ptr + 26, d27);
    _mm256_storeu_si256((__m256i *) ptr + 27, d28);
    _mm256_storeu_si256((__m256i *) ptr + 28, d29);
    _mm256_storeu_si256((__m256i *) ptr + 29, d30);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_31v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    __m256i d17 = _mm256_lddqu_si256((__m256i const *) ptr + 16);
    __m256i d18 = _mm256_lddqu_si256((__m256i const *) ptr + 17);
    __m256i d19 = _mm256_lddqu_si256((__m256i const *) ptr + 18);
    __m256i d20 = _mm256_lddqu_si256((__m256i const *) ptr + 19);
    __m256i d21 = _mm256_lddqu_si256((__m256i const *) ptr + 20);
    __m256i d22 = _mm256_lddqu_si256((__m256i const *) ptr + 21);
    __m256i d23 = _mm256_lddqu_si256((__m256i const *) ptr + 22);
    __m256i d24 = _mm256_lddqu_si256((__m256i const *) ptr + 23);
    __m256i d25 = _mm256_lddqu_si256((__m256i const *) ptr + 24);
    __m256i d26 = _mm256_lddqu_si256((__m256i const *) ptr + 25);
    __m256i d27 = _mm256_lddqu_si256((__m256i const *) ptr + 26);
    __m256i d28 = _mm256_lddqu_si256((__m256i const *) ptr + 27);
    __m256i d29 = _mm256_lddqu_si256((__m256i const *) ptr + 28);
    __m256i d30 = _mm256_lddqu_si256((__m256i const *) ptr + 29);
    __m256i d31 = _mm256_lddqu_si256((__m256i const *) ptr + 30);
    bitonic_sort_uint64_t_31v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 16, d17);
    _mm256_storeu_si256((__m256i *) ptr + 17, d18);
    _mm256_storeu_si256((__m256i *) ptr + 18, d19);
    _mm256_storeu_si256((__m256i *) ptr + 19, d20);
    _mm256_storeu_si256((__m256i *) ptr + 20, d21);
    _mm256_storeu_si256((__m256i *) ptr + 21, d22);
    _mm256_storeu_si256((__m256i *) ptr + 22, d23);
    _mm256_storeu_si256((__m256i *) ptr + 23, d24);
    _mm256_storeu_si256((__m256i *) ptr + 24, d25);
    _mm256_storeu_si256((__m256i *) ptr + 25, d26);
    _mm256_storeu_si256((__m256i *) ptr + 26, d27);
    _mm256_storeu_si256((__m256i *) ptr + 27, d28);
    _mm256_storeu_si256((__m256i *) ptr + 28, d29);
    _mm256_storeu_si256((__m256i *) ptr + 29, d30);
    _mm256_storeu_si256((__m256i *) ptr + 30, d31);
}

static __attribute__((noinline)) void bitonic_sort_uint64_t_32v(uint64_t *ptr) {
    __m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    __m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    __m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    __m256i d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);
    __m256i d05 = _mm256_lddqu_si256((__m256i const *) ptr + 4);
    __m256i d06 = _mm256_lddqu_si256((__m256i const *) ptr + 5);
    __m256i d07 = _mm256_lddqu_si256((__m256i const *) ptr + 6);
    __m256i d08 = _mm256_lddqu_si256((__m256i const *) ptr + 7);
    __m256i d09 = _mm256_lddqu_si256((__m256i const *) ptr + 8);
    __m256i d10 = _mm256_lddqu_si256((__m256i const *) ptr + 9);
    __m256i d11 = _mm256_lddqu_si256((__m256i const *) ptr + 10);
    __m256i d12 = _mm256_lddqu_si256((__m256i const *) ptr + 11);
    __m256i d13 = _mm256_lddqu_si256((__m256i const *) ptr + 12);
    __m256i d14 = _mm256_lddqu_si256((__m256i const *) ptr + 13);
    __m256i d15 = _mm256_lddqu_si256((__m256i const *) ptr + 14);
    __m256i d16 = _mm256_lddqu_si256((__m256i const *) ptr + 15);
    __m256i d17 = _mm256_lddqu_si256((__m256i const *) ptr + 16);
    __m256i d18 = _mm256_lddqu_si256((__m256i const *) ptr + 17);
    __m256i d19 = _mm256_lddqu_si256((__m256i const *) ptr + 18);
    __m256i d20 = _mm256_lddqu_si256((__m256i const *) ptr + 19);
    __m256i d21 = _mm256_lddqu_si256((__m256i const *) ptr + 20);
    __m256i d22 = _mm256_lddqu_si256((__m256i const *) ptr + 21);
    __m256i d23 = _mm256_lddqu_si256((__m256i const *) ptr + 22);
    __m256i d24 = _mm256_lddqu_si256((__m256i const *) ptr + 23);
    __m256i d25 = _mm256_lddqu_si256((__m256i const *) ptr + 24);
    __m256i d26 = _mm256_lddqu_si256((__m256i const *) ptr + 25);
    __m256i d27 = _mm256_lddqu_si256((__m256i const *) ptr + 26);
    __m256i d28 = _mm256_lddqu_si256((__m256i const *) ptr + 27);
    __m256i d29 = _mm256_lddqu_si256((__m256i const *) ptr + 28);
    __m256i d30 = _mm256_lddqu_si256((__m256i const *) ptr + 29);
    __m256i d31 = _mm256_lddqu_si256((__m256i const *) ptr + 30);
    __m256i d32 = _mm256_lddqu_si256((__m256i const *) ptr + 31);
    bitonic_sort_uint64_t_32v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    _mm256_storeu_si256((__m256i *) ptr + 4, d05);
    _mm256_storeu_si256((__m256i *) ptr + 5, d06);
    _mm256_storeu_si256((__m256i *) ptr + 6, d07);
    _mm256_storeu_si256((__m256i *) ptr + 7, d08);
    _mm256_storeu_si256((__m256i *) ptr + 8, d09);
    _mm256_storeu_si256((__m256i *) ptr + 9, d10);
    _mm256_storeu_si256((__m256i *) ptr + 10, d11);
    _mm256_storeu_si256((__m256i *) ptr + 11, d12);
    _mm256_storeu_si256((__m256i *) ptr + 12, d13);
    _mm256_storeu_si256((__m256i *) ptr + 13, d14);
    _mm256_storeu_si256((__m256i *) ptr + 14, d15);
    _mm256_storeu_si256((__m256i *) ptr + 15, d16);
    _mm256_storeu_si256((__m256i *) ptr + 16, d17);
    _mm256_storeu_si256((__m256i *) ptr + 17, d18);
    _mm256_storeu_si256((__m256i *) ptr + 18, d19);
    _mm256_storeu_si256((__m256i *) ptr + 19, d20);
    _mm256_storeu_si256((__m256i *) ptr + 20, d21);
    _mm256_storeu_si256((__m256i *) ptr + 21, d22);
    _mm256_storeu_si256((__m256i *) ptr + 22, d23);
    _mm256_storeu_si256((__m256i *) ptr + 23, d24);
    _mm256_storeu_si256((__m256i *) ptr + 24, d25);
    _mm256_storeu_si256((__m256i *) ptr + 25, d26);
    _mm256_storeu_si256((__m256i *) ptr + 26, d27);
    _mm256_storeu_si256((__m256i *) ptr + 27, d28);
    _mm256_storeu_si256((__m256i *) ptr + 28, d29);
    _mm256_storeu_si256((__m256i *) ptr + 29, d30);
    _mm256_storeu_si256((__m256i *) ptr + 30, d31);
    _mm256_storeu_si256((__m256i *) ptr + 31, d32);
}

static void bitonic_sort_uint64_t(uint64_t *ptr, int length) {
    const int N = 4;

    switch(length / N) {
        case 1: bitonic_sort_uint64_t_01v(ptr); break;
        case 2: bitonic_sort_uint64_t_02v(ptr); break;
        case 3: bitonic_sort_uint64_t_03v(ptr); break;
        case 4: bitonic_sort_uint64_t_04v(ptr); break;
        case 5: bitonic_sort_uint64_t_05v(ptr); break;
        case 6: bitonic_sort_uint64_t_06v(ptr); break;
        case 7: bitonic_sort_uint64_t_07v(ptr); break;
        case 8: bitonic_sort_uint64_t_08v(ptr); break;
        case 9: bitonic_sort_uint64_t_09v(ptr); break;
        case 10: bitonic_sort_uint64_t_10v(ptr); break;
        case 11: bitonic_sort_uint64_t_11v(ptr); break;
        case 12: bitonic_sort_uint64_t_12v(ptr); break;
        case 13: bitonic_sort_uint64_t_13v(ptr); break;
        case 14: bitonic_sort_uint64_t_14v(ptr); break;
        case 15: bitonic_sort_uint64_t_15v(ptr); break;
        case 16: bitonic_sort_uint64_t_16v(ptr); break;
        case 17: bitonic_sort_uint64_t_17v(ptr); break;
        case 18: bitonic_sort_uint64_t_18v(ptr); break;
        case 19: bitonic_sort_uint64_t_19v(ptr); break;
        case 20: bitonic_sort_uint64_t_20v(ptr); break;
        case 21: bitonic_sort_uint64_t_21v(ptr); break;
        case 22: bitonic_sort_uint64_t_22v(ptr); break;
        case 23: bitonic_sort_uint64_t_23v(ptr); break;
        case 24: bitonic_sort_uint64_t_24v(ptr); break;
        case 25: bitonic_sort_uint64_t_25v(ptr); break;
        case 26: bitonic_sort_uint64_t_26v(ptr); break;
        case 27: bitonic_sort_uint64_t_27v(ptr); break;
        case 28: bitonic_sort_uint64_t_28v(ptr); break;
        case 29: bitonic_sort_uint64_t_29v(ptr); break;
        case 30: bitonic_sort_uint64_t_30v(ptr); break;
        case 31: bitonic_sort_uint64_t_31v(ptr); break;
        case 32: bitonic_sort_uint64_t_32v(ptr); break;
    }
}
}
}
