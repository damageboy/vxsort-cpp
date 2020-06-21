#ifndef GCSORT_BM_FULLSORT_H
#define GCSORT_BM_FULLSORT_H

#include <random>
#include <algorithm>
#include <thread>
#include <benchmark/benchmark.h>
#include "stolen-cycleclock.h"
#include "util.h"

#include <vxsort.h>

using gcsort::vector_machine;

namespace vxsort_bench {
const auto processor_count = 1;

static const int MIN_SORT = 256;
static const int MAX_SORT = 1 << 20;

template <class Q>
static void BM_stdsort(benchmark::State& state) {
  auto n = state.range(0);
  auto v = std::vector<Q>(n);
  const auto ITERATIONS = 10;

  generate_unique_ptrs_vec(v, n);
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

template <class Q, gcsort::vector_machine M, int U>
static void BM_vxsort(benchmark::State& state) {
  auto n = state.range(0);
  auto v = std::vector<Q>(n);
  const auto ITERATIONS = 10;

  generate_unique_ptrs_vec(v, n);
  auto copies = generate_copies(ITERATIONS, n, v);
  auto begins = generate_array_beginnings(copies);
  auto ends = generate_array_beginnings(copies);
  for (size_t i = 0; i < copies.size(); i++)
    ends[i] = begins[i] + n - 1;

  auto sorter = gcsort::vxsort<Q, M, U>();

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

#endif  // GCSORT_BM_FULLSORT_H
