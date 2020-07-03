#include <random>
#include <algorithm>
#include <thread>
#include <benchmark/benchmark.h>

#include <introsort_orig.h>
#include <isa_detection.h>
#include <smallsort/bitonic_sort.h>

#include "../stolen-cycleclock.h"
#include "../util.h"

using vxsort::vector_machine;

namespace vxsort_bench {

const auto processor_count = std::thread::hardware_concurrency();

static void BM_insertionsort(benchmark::State& state) {
  static const int ITERATIONS = 1024;
  auto n = state.range(0);
  auto v = std::vector<int64_t>(n);
  generate_unique_ptrs_vec(v, (int64_t)0x1000, (int64_t) 8);

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
    for (auto i = 0; i < ITERATIONS; i++)
      sort_insertionsort((uint8_t**)begins[i], (uint8_t**)ends[i]);
    total_cycles += cycleclock::Now() - start;
  }

  state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
  state.counters["Cycles/N"] = make_cycle_per_n_counter(
      (double)total_cycles / (double)(n * ITERATIONS * state.iterations()));
}

template <class Q, vxsort::vector_machine M>
static void BM_bitonic_sort(benchmark::State& state) {
  if (!vxsort::supports_vector_machine(M)) {
    state.SkipWithError(
        "Current CPU does not support the minimal features for this test");
    return;
  }

  static const int ITERATIONS = 1024;
  auto n = state.range(0);
  auto v = std::vector<Q>(n);
  generate_unique_ptrs_vec(v, (Q) 0x1000, (Q) 0x8);

  auto copies = generate_copies(ITERATIONS, n, v);
  auto begins = generate_array_beginnings(copies);

  uint64_t total_cycles = 0;
  for (auto _ : state) {
    state.PauseTiming();
    refresh_copies(copies, v);
    state.ResumeTiming();
    auto start = cycleclock::Now();
    for (auto i = 0; i < ITERATIONS; i++)
      vxsort::smallsort::bitonic<Q, M>::sort(begins[i], n);
    total_cycles += cycleclock::Now() - start;
  }

  state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
  state.counters["Cycles/N"] = make_cycle_per_n_counter(
      (double)total_cycles / (double)(n * ITERATIONS * state.iterations()));
}

template <class Q, vxsort::vector_machine M>
static void BM_bitonic_sort_alt(benchmark::State& state) {
    if (!vxsort::supports_vector_machine(M)) {
        state.SkipWithError(
            "Current CPU does not support the minimal features for this test");
        return;
    }

    static const int ITERATIONS = 1024;
    auto n = state.range(0);
    auto v = std::vector<Q>(n);
    generate_unique_ptrs_vec(v, (Q) 0x1000, (Q) 0x8);

    auto copies = generate_copies(ITERATIONS, n, v);
    auto begins = generate_array_beginnings(copies);

    uint64_t total_cycles = 0;
    for (auto _ : state) {
        state.PauseTiming();
        refresh_copies(copies, v);
        state.ResumeTiming();
        auto start = cycleclock::Now();
        for (auto i = 0; i < ITERATIONS; i++)
            vxsort::smallsort::bitonic<Q, M>::sort_alt(begins[i], n);
        total_cycles += cycleclock::Now() - start;
    }

    state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
    state.counters["Cycles/N"] = make_cycle_per_n_counter(
        (double)total_cycles / (double)(n * ITERATIONS * state.iterations()));
}

}
