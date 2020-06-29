#include <random>
#include <algorithm>
#include <thread>
#include <benchmark/benchmark.h>

#include <isa_detection.h>
#include <packer.h>

#include "../stolen-cycleclock.h"
#include "../util.h"

using vxsort::vector_machine;

namespace vxsort_bench {

//const auto processor_count = std::thread::hardware_concurrency();
const auto processor_count = 1;

static const int MIN_PACKUNPACK = 256;
static const int MAX_PACKUNPACK = 1 << 20;


template <typename Q1, typename Q2, vxsort::vector_machine M, int Shift, int Unroll>
static void BM_pack(benchmark::State& state) {
  if (!vxsort::supports_vector_machine(M)) {
    state.SkipWithError(
        "Current CPU does not support the minimal features for this test");
    return;
  }

  static const int ITERATIONS = 10;
  auto n = state.range(0);
  auto v = std::vector<Q1>(n);
  generate_unique_ptrs_vec(v, (Q1) 0x1000L, (Q1) 1<<Shift);

  auto copies = generate_copies(ITERATIONS, n, v);
  auto begins = generate_array_beginnings(copies);

  uint64_t total_cycles = 0;
  for (auto _ : state) {
    state.PauseTiming();
    refresh_copies(copies, v);
    state.ResumeTiming();
    auto start = cycleclock::Now();
    for (auto i = 0; i < ITERATIONS; i++)
      vxsort::packer<Q1, Q2, M, Shift, Unroll, MIN_PACKUNPACK>::pack(begins[i], n, 0x1000);
    total_cycles += cycleclock::Now() - start;
  }

  state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
  state.counters["Cycles/N"] = make_cycle_per_n_counter(
      (double)total_cycles / (double)(n * ITERATIONS * state.iterations()));
}

template <typename Q1, typename Q2, vxsort::vector_machine M, int Shift, int Unroll>
static void BM_unpack(benchmark::State& state) {
    if (!vxsort::supports_vector_machine(M)) {
        state.SkipWithError(
            "Current CPU does not support the minimal features for this test");
        return;
    }

    static const int ITERATIONS = 10;
    auto n = state.range(0);
    auto v = std::vector<Q1>(n);
    generate_unique_ptrs_vec(v, (Q1) 0x1000L, (Q1) 1<<Shift);

    // Do one packing
    vxsort::packer<Q1, Q2, M, Shift, Unroll, MIN_PACKUNPACK>::pack(v.data(), n, 0x1000);

    auto copies = generate_copies(ITERATIONS, n, v);
    auto begins = generate_array_beginnings(copies);

    uint64_t total_cycles = 0;
    for (auto _ : state) {
        state.PauseTiming();
        refresh_copies(copies, v);
        state.ResumeTiming();
        auto start = cycleclock::Now();
        for (auto i = 0; i < ITERATIONS; i++)
            vxsort::packer<Q1, Q2, M, Shift, Unroll, MIN_PACKUNPACK>::unpack((Q2 *) begins[i], n, 0x1000);
        total_cycles += cycleclock::Now() - start;
    }

    state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
    state.counters["Cycles/N"] = make_cycle_per_n_counter(
        (double)total_cycles / (double)(n * ITERATIONS * state.iterations()));
}
}
