#include <random>
#include <algorithm>
#include <benchmark/benchmark.h>

#include <introsort.h>
#include <smallsort/bitonic_sort.int32_t.generated.h>
#include <smallsort/bitonic_sort.int64_t.generated.h>
#include <vxsort.h>
#include "util.h"

benchmark::Counter make_time_per_n_counter(int64_t n) {
  return benchmark::Counter(
      (double) n,
      benchmark::Counter::kIsIterationInvariantRate |
      benchmark::Counter::kInvert,
      benchmark::Counter::kIs1000);
}

static void BM_full_introsort(benchmark::State &state) {
    auto n = state.range(0);
    auto v = std::vector<uint8_t*>(n);
    const auto ITERATIONS = 10;
    generate_unique_ptrs_vec(v, n);
    auto copies = generate_copies(ITERATIONS, n, v);
    auto begins = generate_array_beginnings(copies);
    auto ends = generate_array_beginnings(copies);
    for (auto i = 0; i < copies.size(); i++)
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

static const int MIN_SORT = 4096;
static const int MAX_SORT = 1 << 20;

BENCHMARK(BM_full_introsort)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond);

template <class Q, int U> static void BM_full_vxsort(benchmark::State &state) {
    auto n = state.range(0);
    auto v = std::vector<Q>(n);
    const auto ITERATIONS = 10;

    generate_unique_ptrs_vec(v, n);
    auto copies = generate_copies(ITERATIONS, n, v);
    auto begins = generate_array_beginnings(copies);
    auto ends = generate_array_beginnings(copies);
    for (auto i = 0; i < copies.size(); i++)
        ends[i] = begins[i] + n - 1;

    auto sorter = gcsort::vxsort<Q, U>();

    for (auto _ : state) {
        state.PauseTiming();
        refresh_copies(copies, v);
        state.ResumeTiming();
        for (auto i = 0; i < ITERATIONS; i++) {
            sorter.sort(begins[i], ends[i]);
        }
    }

    state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
}
BENCHMARK_TEMPLATE(BM_full_vxsort, int64_t, 1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_full_vxsort, int64_t, 4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_full_vxsort, int64_t, 8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_full_vxsort, int64_t, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_full_vxsort, int32_t, 1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_full_vxsort, int32_t, 4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_full_vxsort, int32_t, 8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond);

static void BM_insertionsort(benchmark::State &state) {
  static const int ITERATIONS = 1024;
  auto n = state.range(0);
  auto v = std::vector<int64_t>(n);
  generate_unique_ptrs_vec(v, n);

  auto copies = generate_copies(ITERATIONS, n, v);
  auto begins = generate_array_beginnings(copies);
  auto ends = generate_array_beginnings(copies);
  for (auto i = 0; i < copies.size(); i++)
    ends[i] = begins[i] + n - 1;

  for (auto _ : state) {
    state.PauseTiming();
    refresh_copies(copies, v);
    state.ResumeTiming();
    for (auto i = 0; i < ITERATIONS; i++)
      sort_insertionsort((uint8_t **) begins[i], (uint8_t **) ends[i]);
  }

  delete_copies(copies);

  state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
}
BENCHMARK(BM_insertionsort)->DenseRange(4, 64, 4)->Unit(benchmark::kNanosecond);


static void BM_bitonic_sort_int64(benchmark::State &state) {
  static const int ITERATIONS = 1024;
  auto n = state.range(0);
  auto v = std::vector<int64_t>(n);
  generate_unique_ptrs_vec(v, n);

  auto copies = generate_copies(ITERATIONS, n, v);
  auto begins = generate_array_beginnings(copies);

  for (auto _ : state) {
    state.PauseTiming();
    refresh_copies(copies, v);
    state.ResumeTiming();
    for (auto i = 0; i < ITERATIONS; i++)
      gcsort::smallsort::bitonic<int64_t>::sort(begins[i], n);
  }


  delete_copies(copies);

  state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
}
BENCHMARK(BM_bitonic_sort_int64)->DenseRange(4, 64, 4)->Unit(benchmark::kNanosecond);


static void BM_bitonic_sort_int32(benchmark::State &state) {
  static const int ITERATIONS = 1024;
  auto n = state.range(0);
  auto v = std::vector<int32_t>(n);
  generate_unique_ptrs_vec(v, n);

  auto copies = generate_copies(ITERATIONS, n, v);
  auto begins = generate_array_beginnings(copies);

  for (auto _ : state) {
    state.PauseTiming();
    refresh_copies(copies, v);
    state.ResumeTiming();
    for (auto i = 0; i < ITERATIONS; i++)
      gcsort::smallsort::bitonic<int32_t>::sort(begins[i], n);
  }

  delete_copies(copies);

  state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
}
BENCHMARK(BM_bitonic_sort_int32)->DenseRange(8, 128, 8)->Unit(benchmark::kNanosecond);
