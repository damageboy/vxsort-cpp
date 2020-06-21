#include <random>
#include <algorithm>
#include <thread>
#include <benchmark/benchmark.h>

#include <introsort_orig.h>
#include <smallsort/bitonic_sort.AVX2.double.generated.h>
#include <smallsort/bitonic_sort.AVX2.float.generated.h>
#include <smallsort/bitonic_sort.AVX2.int32_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.int64_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.uint32_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.uint64_t.generated.h>
#include <smallsort/bitonic_sort.AVX512.float.generated.h>
#include <smallsort/bitonic_sort.AVX512.int32_t.generated.h>
#include <smallsort/bitonic_sort.AVX512.uint32_t.generated.h>

#include "util.h"

using gcsort::vector_machine;

namespace vxsort_bench {

const auto processor_count = std::thread::hardware_concurrency();

static void BM_insertionsort(benchmark::State &state) {
  static const int ITERATIONS = 1024;
  auto n = state.range(0);
  auto v = std::vector<int64_t>(n);
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
    for (auto i = 0; i < ITERATIONS; i++)
      sort_insertionsort((uint8_t **) begins[i], (uint8_t **) ends[i]);
  }

  state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
}
BENCHMARK(BM_insertionsort)->DenseRange(4, 64, 4)->Unit(benchmark::kNanosecond);


template <class Q, gcsort::vector_machine M>
static void BM_bitonic_sort(benchmark::State &state) {
  static const int ITERATIONS = 1024;
  auto n = state.range(0);
  auto v = std::vector<Q>(n);
  generate_unique_ptrs_vec(v, n);

  auto copies = generate_copies(ITERATIONS, n, v);
  auto begins = generate_array_beginnings(copies);

  for (auto _ : state) {
    state.PauseTiming();
    refresh_copies(copies, v);
    state.ResumeTiming();
    for (auto i = 0; i < ITERATIONS; i++)
      gcsort::smallsort::bitonic<Q, M>::sort(begins[i], n);
  }

  state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
}

BENCHMARK_TEMPLATE(BM_bitonic_sort, int32_t,  vector_machine::AVX2)->DenseRange(8, 128, 8)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, uint32_t, vector_machine::AVX2)->DenseRange(8, 128, 8)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, float,    vector_machine::AVX2)->DenseRange(8, 128, 8)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, int64_t,  vector_machine::AVX2)->DenseRange(4,  64, 4)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, uint64_t, vector_machine::AVX2)->DenseRange(4,  64, 4)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, double,   vector_machine::AVX2)->DenseRange(4,  64, 4)->Unit(benchmark::kNanosecond);

BENCHMARK_TEMPLATE(BM_bitonic_sort, int32_t,  vector_machine::AVX512)->DenseRange(16, 512, 16)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, uint32_t, vector_machine::AVX512)->DenseRange(16, 512, 16)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, float,    vector_machine::AVX512)->DenseRange(16, 512, 16)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, int64_t,  vector_machine::AVX512)->DenseRange( 8, 256,  8)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, uint64_t, vector_machine::AVX512)->DenseRange( 8, 256,  8)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, double,   vector_machine::AVX512)->DenseRange( 8, 256,  8)->Unit(benchmark::kNanosecond);

}

