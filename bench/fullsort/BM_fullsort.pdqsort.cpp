#include "vxsort_targets_enable_avx2.h"

#include <random>
#include <benchmark/benchmark.h>

#include "BM_fullsort.h"
#include "../reference/introsort/introsort_orig.h"

namespace vxsort_bench {
using namespace vxsort::types;

static void BM_full_introsort(benchmark::State& state) {
    auto n = state.range(0);
    auto v = std::vector<i64>(n);
    const auto ITERATIONS = 10;
    generate_unique_values_vec(v, (i64)0x1000, (i64)8);
    auto copies = generate_copies(ITERATIONS, n, v);
    auto begins = generate_array_beginnings(copies);
    auto ends = generate_array_beginnings(copies);
    for (usize i = 0; i < copies.size(); i++)
        ends[i] = begins[i] + n - 1;

    u64 total_cycles = 0;
    for (auto _ : state) {
        state.PauseTiming();
        refresh_copies(copies, v);
        state.ResumeTiming();
        auto start = cycleclock::Now();
        for (auto i = 0; i < ITERATIONS; i++) {
            sort_introsort((u8**)begins[i], (u8**)ends[i]);
        }
        total_cycles += (cycleclock::Now() - start);
    }

    state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
    process_perf_counters(state.counters, n * ITERATIONS);
    if (!state.counters.contains("cycles/N"))
        state.counters["rdtsc-cycles/N"] = make_cycle_per_n_counter((f64)total_cycles / (f64)(n * ITERATIONS * state.iterations()));
}

BENCHMARK(BM_full_introsort)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_stdsort, i32)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_stdsort, u32)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_stdsort, f32)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_stdsort, i64)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_stdsort, u64)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_stdsort, f64)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
}

#include "vxsort_targets_disable.h"
