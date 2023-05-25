#include "vxsort_targets_enable_avx2.h"

#include <random>
#include <benchmark/benchmark.h>

#include "fullsort_params.h"
#include "../util.h"
#include "../reference/pdqsort.h"

namespace vxsort_bench {
using namespace vxsort::types;

template <class Q>
static void BM_pdqsort_branchless(benchmark::State& state) {
    auto n = state.range(0);
    const auto ITERATIONS = 10;

    auto v = generate_unique_values_vec(n, (Q)0x1000, (Q)8);
    auto copies = generate_copies(ITERATIONS, n, v);
    auto begins = generate_array_beginnings(copies);
    auto ends = generate_array_beginnings(copies);
    for (usize i = 0; i < copies.size(); i++)
        ends[i] = begins[i] + n - 1;

    vxsort::u64 total_cycles = 0;
    for (auto _ : state) {
        state.PauseTiming();
        refresh_copies(copies, v);
        state.ResumeTiming();
        auto start = cycleclock::Now();
        for (auto i = 0; i < ITERATIONS; i++) {
            pdqsort_branchless(begins[i], ends[i]);
        }
        total_cycles += (cycleclock::Now() - start);
    }

    state.SetLabel(get_crypto_hash(begins[0], ends[0]));
    state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
    state.SetBytesProcessed(state.iterations() * n * ITERATIONS * sizeof(Q));
    process_perf_counters(state.counters, n * ITERATIONS);
    if (!state.counters.contains("cycles/N"))
        state.counters["rdtsc-cycles/N"] = make_cycle_per_n_counter((f64)total_cycles / (f64)(n * ITERATIONS * state.iterations()));
}

BENCHMARK_TEMPLATE(BM_pdqsort_branchless, i16)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_pdqsort_branchless, u16)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_pdqsort_branchless, i32)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_pdqsort_branchless, u32)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_pdqsort_branchless, f32)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_pdqsort_branchless, i64)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_pdqsort_branchless, u64)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_pdqsort_branchless, f64)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
}

#include "vxsort_targets_disable.h"
