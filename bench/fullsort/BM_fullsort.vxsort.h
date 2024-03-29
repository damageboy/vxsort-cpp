#ifndef VXSORT_BM_FULLSORT_VXSORT_H
#define VXSORT_BM_FULLSORT_VXSORT_H

#include <benchmark/benchmark.h>
#include <algorithm>
#include <random>
#include <thread>
#include "../util.h"
#include "../bench_isa.h"

#include <vxsort.h>

#include "fullsort_params.h"

namespace vxsort_bench {
using namespace vxsort::types;
using vxsort::vector_machine;

template <class Q, vector_machine M, i32 U>
static void BM_vxsort(benchmark::State& state) {
    VXSORT_BENCH_ISA();

    auto n = state.range(0);
    auto v = std::vector<Q>((i32)n);
    const auto ITERATIONS = 10;

    generate_unique_values_vec(v, (Q)0x1000, (Q)0x8);
    auto copies = generate_copies(ITERATIONS, n, v);
    auto begins = generate_array_beginnings(copies);
    auto ends = generate_array_beginnings(copies);
    for (usize i = 0; i < copies.size(); i++)
        ends[i] = begins[i] + n - 1;

    auto sorter = ::vxsort::vxsort<Q, M, U>();

    u64 total_cycles = 0;
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

    state.SetLabel(get_crypto_hash(begins[0], ends[0]));
    state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
    state.SetBytesProcessed(state.iterations() * n * ITERATIONS * sizeof(Q));
    process_perf_counters(state.counters, n * ITERATIONS);
    if (!state.counters.contains("cycles/N"))
        state.counters["rdtsc-cycles/N"] = make_cycle_per_n_counter((f64)total_cycles / (f64)(n * ITERATIONS * state.iterations()));
}

const i32 StridedSortSize = 1000000;
const i64 StridedSortMinValue = 0x80000000LL;

template <class Q, vector_machine M, i32 U>
static void BM_vxsort_strided(benchmark::State& state) {
    VXSORT_BENCH_ISA();

    auto n = StridedSortSize;
    auto stride = state.range(0);
    auto v = std::vector<Q>(n);
    const auto ITERATIONS = 10;

    const auto min_value = StridedSortMinValue;
    const auto max_value = min_value + StridedSortSize * stride;

    generate_unique_values_vec(v, (Q) 0x80000000, (Q) stride);
    auto copies = generate_copies(ITERATIONS, n, v);
    auto begins = generate_array_beginnings(copies);
    auto ends = generate_array_beginnings(copies);
    for (size_t i = 0; i < copies.size(); i++)
        ends[i] = begins[i] + n - 1;

    auto sorter = ::vxsort::vxsort<Q, M, U, 3>();

    u64 total_cycles = 0;
    for (auto _ : state) {
        state.PauseTiming();
        refresh_copies(copies, v);
        state.ResumeTiming();
        auto start = cycleclock::Now();
        for (auto i = 0; i < ITERATIONS; i++) {
            sorter.sort(begins[i], ends[i], min_value, max_value);
        }
        total_cycles += (cycleclock::Now() - start);
    }

    state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
    process_perf_counters(state.counters, n * ITERATIONS);
    if (!state.counters.contains("cycles/N"))
        state.counters["rdtsc-cycles/N"] = make_cycle_per_n_counter((f64)total_cycles / (f64)(n * ITERATIONS * state.iterations()));
}
}

#endif  // VXSORT_BM_FULLSORT_VXSORT_H
