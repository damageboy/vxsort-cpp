#include "vxsort_targets_enable_avx2.h"

#include "BM_smallsort.h"

#include <smallsort/bitonic_sort.avx2.h>

namespace vxsort_bench {
using namespace vxsort::types;
using benchmark::TimeUnit;
using vm = vxsort::vector_machine;

#define COEX(a, b){                     \
    auto vec_tmp = a;                   \
    a = _mm256_min_epi32(a, b);         \
    b = _mm256_max_epi32(vec_tmp, b);}

/* shuffle 2 vectors, instruction for int is missing,
 * therefore shuffle with float */
#define SHUFFLE_2_VECS(a, b, mask)                                       \
    _mm256_castps_si256 (_mm256_shuffle_ps(                         \
        _mm256_castsi256_ps (a), _mm256_castsi256_ps (b), mask));

/* optimized sorting network for two vectors, that is 16 ints */
inline void sort_02v_ascending(__m256i &v1, __m256i &v2) {
    COEX(v1, v2);                                  /* step 1 */

    v2 = _mm256_shuffle_epi32(v2, _MM_SHUFFLE(2, 3, 0, 1)); /* step 2 */
    COEX(v1, v2);

    auto tmp = v1;                                          /* step  3 */
    v1 = SHUFFLE_2_VECS(v1, v2, 0b10001000);
    v2 = SHUFFLE_2_VECS(tmp, v2, 0b11011101);
    COEX(v1, v2);

    v2 = _mm256_shuffle_epi32(v2, _MM_SHUFFLE(0, 1, 2, 3)); /* step  4 */
    COEX(v1, v2);

    tmp = v1;                                               /* step  5 */
    v1 = SHUFFLE_2_VECS(v1, v2, 0b01000100);
    v2 = SHUFFLE_2_VECS(tmp, v2, 0b11101110);
    COEX(v1, v2);

    tmp = v1;                                               /* step  6 */
    v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
    v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
    COEX(v1, v2);

    v2 = _mm256_permutevar8x32_epi32(v2, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
    COEX(v1, v2);                                           /* step  7 */

    tmp = v1;                                               /* step  8 */
    v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
    v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
    COEX(v1, v2);

    tmp = v1;                                               /* step  9 */
    v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
    v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
    COEX(v1, v2);

    /* permute to make it easier to restore order */
    v1 = _mm256_permutevar8x32_epi32(v1, _mm256_setr_epi32(0, 4, 1, 5, 6, 2, 7, 3));
    v2 = _mm256_permutevar8x32_epi32(v2, _mm256_setr_epi32(0, 4, 1, 5, 6, 2, 7, 3));

    tmp = v1;                                              /* step  10 */
    v1 = SHUFFLE_2_VECS(v1, v2, 0b10001000);
    v2 = SHUFFLE_2_VECS(tmp, v2, 0b11011101);
    COEX(v1, v2);

    /* restore order */
    auto b2 = _mm256_shuffle_epi32(v2, 0b10110001);
    auto b1 = _mm256_shuffle_epi32(v1, 0b10110001);
    v1 = _mm256_blend_epi32(v1, b2, 0b10101010);
    v2 = _mm256_blend_epi32(b1, v2, 0b10101010);
}

// This is generated for testing purposes only
void bitonic_blacher_16_i32(i32 *ptr) {
    auto d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
    auto d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);;
    sort_02v_ascending(d01, d02);
    _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    _mm256_storeu_si256((__m256i *) ptr + 1, d02);
}


void BM_blacher(benchmark::State& state)
{
    if (!vxsort::supports_vector_machine(vector_machine::AVX2)) {
        state.SkipWithError("Current CPU does not support the minimal features for this test");
        return;
    }

    static const i32 ITERATIONS = 1024;
    auto n = 16;
    auto v = generate_unique_values_vec(n, (i32)0x1000, (i32)0x8);

    auto copies = generate_copies(ITERATIONS, n, v);
    auto begins = generate_array_beginnings(copies);

    uint64_t total_cycles = 0;
    for (auto _ : state) {
        state.PauseTiming();
        refresh_copies(copies, v);
        state.ResumeTiming();
        auto start = cycleclock::Now();
        for (auto i = 0; i < ITERATIONS; i++) {
            bitonic_blacher_16_i32(begins[i]);
        }
        total_cycles += cycleclock::Now() - start;
    }

    state.SetBytesProcessed(state.iterations() * n * ITERATIONS * sizeof(i32));

    state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
    process_perf_counters(state.counters, n * ITERATIONS);
    if (!state.counters.contains("cycles/N"))
        state.counters["rdtsc-cycles/N"] = make_cycle_per_n_counter((f64)total_cycles / (f64)(n * ITERATIONS * state.iterations()));
}

BENCHMARK(BM_blacher)->Unit(kNanosecond)->MinTime(0.1);

}

#include "vxsort_targets_disable.h"
