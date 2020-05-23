
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
    }
}
}
}
