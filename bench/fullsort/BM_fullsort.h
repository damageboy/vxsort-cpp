#ifndef VXSORT_BM_FULLSORT_H
#define VXSORT_BM_FULLSORT_H

#include <benchmark/benchmark.h>
#include <algorithm>
#include <random>
#include <thread>
#include "../stolen-cycleclock.h"
#include "../util.h"

#include <vxsort.h>

using vxsort::vector_machine;

namespace vxsort_bench {
const auto processor_count = 1;

static const int MIN_SORT = 256;
static const int MAX_SORT = 1 << 20;

template <class Q>
static void BM_stdsort(benchmark::State& state) {
  auto n = state.range(0);
  auto v = std::vector<Q>(n);
  const auto ITERATIONS = 10;

  generate_unique_ptrs_vec(v, (Q) 0x1000, (Q) 8);
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
      std::sort(begins[i], ends[i]);
    }
    total_cycles += (cycleclock::Now() - start);
  }

  state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
  state.counters["Cycles/N"] = make_cycle_per_n_counter((double) total_cycles / (double) (n * ITERATIONS * state.iterations()));
}

template <class Q, vxsort::vector_machine M, int U>
static void BM_vxsort(benchmark::State& state) {
  if (!vxsort::supports_vector_machine(M)) {
    state.SkipWithError(
        "Current CPU does not support the minimal features for this test");
    return;
  }

  auto n = state.range(0);
  auto v = std::vector<Q>(n);
  const auto ITERATIONS = 10;

  generate_unique_ptrs_vec(v, (Q) 0x1000, (Q) 0x8);
  auto copies = generate_copies(ITERATIONS, n, v);
  auto begins = generate_array_beginnings(copies);
  auto ends = generate_array_beginnings(copies);
  for (size_t i = 0; i < copies.size(); i++)
    ends[i] = begins[i] + n - 1;

  auto sorter = vxsort::vxsort<Q, M, U>();

  uint64_t total_cycles = 0;
  for (auto _ : state) {
    state.PauseTiming();
    refresh_copies(copies, v);
    state.ResumeTiming();
    auto start = cycleclock::Now();
    for (auto i = 0; i < ITERATIONS; i++) {
      sorter.sort(begins[i], ends[i]);
    }
    total_cycles += (cycleclock::Now() - start);
  }

  state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
  state.counters["Cycles/N"] = make_cycle_per_n_counter((double) total_cycles / (double) (n * ITERATIONS * state.iterations()));
}
}

#endif  // VXSORT_BM_FULLSORT_H
