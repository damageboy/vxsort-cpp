#include <random>
#include <benchmark/benchmark.h>

#include "vxsort_targets_enable_avx2.h"

#include "BM_fullsort.h"

#include "../reference/introsort/introsort_orig.h"

using vxsort::vector_machine;

namespace vxsort_bench {

static void BM_full_introsort(benchmark::State &state) {
  auto n = state.range(0);
  auto v = std::vector<int64_t>(n);
  const auto ITERATIONS = 10;
  generate_unique_ptrs_vec(v, (int64_t) 0x1000, (int64_t) 8);
  auto copies = generate_copies(ITERATIONS, n, v);
  auto begins = generate_array_beginnings(copies);
  auto ends = generate_array_beginnings(copies);
  for (size_t i = 0; i < copies.size(); i++)
    ends[i] = begins[i] + n - 1;

  uint64_t total_cycles = 0;
  for (auto _ : state) {
    state.PauseTiming();
    refresh_copies(copies, v);
    state.ResumeTiming();
    auto start = cycleclock::Now();
    for (auto i = 0; i < ITERATIONS; i++) {
      sort_introsort((uint8_t **) begins[i], (uint8_t **) ends[i]);
    }
    total_cycles += (cycleclock::Now() - start);

  }

  state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
  state.counters["Cycles/N"] = make_cycle_per_n_counter((double) total_cycles / (double) (n * ITERATIONS * state.iterations()));

}

BENCHMARK(BM_full_introsort)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_stdsort, int32_t )->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_stdsort, uint32_t)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_stdsort, float   )->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_stdsort, int64_t )->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_stdsort, uint64_t)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_stdsort, double  )->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
}

#include "vxsort_targets_disable.h"
