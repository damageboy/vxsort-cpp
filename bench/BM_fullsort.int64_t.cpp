#include <random>
#include <benchmark/benchmark.h>

#include "vxsort_targets_enable.h"

#include <introsort_orig.h>

#include "BM_fullsort.h"

#include <vxsort.int64_t.h>

using gcsort::vector_machine;

namespace vxsort_bench {

static void BM_full_introsort(benchmark::State &state) {
    auto n = state.range(0);
    auto v = std::vector<uint8_t*>(n);
    const auto ITERATIONS = 10;
    generate_unique_ptrs_vec(v, n);
    auto copies = generate_copies(ITERATIONS, n, v);
    auto begins = generate_array_beginnings(copies);
    auto ends = generate_array_beginnings(copies);
    for (size_t i = 0; i < copies.size(); i++)
        ends[i] = begins[i] + n - 1;

    for (auto _ : state) {
        state.PauseTiming();
        refresh_copies(copies, v);
        state.ResumeTiming();
        for (auto i = 0; i < ITERATIONS; i++) {
            sort_introsort(begins[i], ends[i]);
        }
    }

  state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
}

BENCHMARK(BM_full_introsort)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_stdsort, int64_t)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_vxsort, int64_t, vector_machine::AVX2,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, int64_t, vector_machine::AVX2,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, int64_t, vector_machine::AVX2,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, int64_t, vector_machine::AVX2, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_vxsort, int64_t, vector_machine::AVX512,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, int64_t, vector_machine::AVX512,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, int64_t, vector_machine::AVX512,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, int64_t, vector_machine::AVX512, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

}

#include "vxsort_targets_disable.h"