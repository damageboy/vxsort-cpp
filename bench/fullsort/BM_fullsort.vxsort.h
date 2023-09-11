#ifndef VXSORT_BM_FULLSORT_VXSORT_H
#define VXSORT_BM_FULLSORT_VXSORT_H

#include <benchmark/benchmark.h>
#include <fmt/format.h>
#include <algorithm>
#include <magic_enum.hpp>
#include <random>
#include <thread>
#include "../bench_isa.h"
#include "../util.h"

#ifndef VXSORT_COMPILER_MSVC
#include <cxxabi.h>
#endif

#include <vxsort.h>

#include "fullsort_params.h"

namespace vxsort_bench {
using namespace vxsort::types;
using benchmark::TimeUnit;
using vxsort::vector_machine;

enum class SortPattern {
    unique_values,
    shuffled_16_values,
    all_equal,
    ascending_int,
    descending_int,
    pipe_organ,
    push_front,
    push_middle
};

template <class Q>
std::vector<Q> generate_pattern(SortPattern pattern, usize size, Q start, Q stride) {
    switch (pattern) {
        case SortPattern::unique_values:
            return unique_values(size, start, stride);
        case SortPattern::shuffled_16_values:
            return shuffled_16_values(size, start, stride);
        case SortPattern::all_equal:
            return all_equal(size, start, stride);
        case SortPattern::ascending_int:
            return ascending_int(size, start, stride);
        case SortPattern::descending_int:
            return descending_int(size, start, stride);
        case SortPattern::pipe_organ:
            return pipe_organ(size, start, stride);
        case SortPattern::push_front:
            return push_front(size, start, stride);
        case SortPattern::push_middle:
            return push_middle(size, start, stride);
        default:
            return unique_values(size, start, stride);
    }
}

template <class Q, vector_machine M, i32 U>
static void BM_vxsort(benchmark::State& state) {
    VXSORT_BENCH_ISA();

    auto n = state.range(0);
    const auto ITERATIONS = 10;

    auto v = unique_values(n, (Q)0x1000, (Q)0x8);
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
        state.counters["rdtsc-cycles/N"] = make_cycle_per_n_counter(
            (f64)total_cycles / (f64)(n * ITERATIONS * state.iterations()));
}

template <class Q, vector_machine M, i32 U>
static void BM_vxsort_pattern(benchmark::State& state, i64 n, SortPattern pattern) {
    VXSORT_BENCH_ISA();

    auto v = generate_pattern(pattern, n, (Q)0x1000, (Q)0x8);

    const auto ITERATIONS = 10;

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
        state.counters["rdtsc-cycles/N"] = make_cycle_per_n_counter(
            (f64)total_cycles / (f64)(n * ITERATIONS * state.iterations()));
}

const i32 StridedSortSize = 1000000;
const i64 StridedSortMinValue = 0x80000000LL;

template <class Q, vector_machine M, i32 U>
static void BM_vxsort_strided(benchmark::State& state) {
    VXSORT_BENCH_ISA();

    auto n = StridedSortSize;
    auto stride = state.range(0);
    const auto ITERATIONS = 10;

    const auto min_value = StridedSortMinValue;
    const auto max_value = min_value + StridedSortSize * stride;

    auto v = unique_values(n, (Q)0x80000000, (Q)stride);
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
        state.counters["rdtsc-cycles/N"] = make_cycle_per_n_counter(
            (f64)total_cycles / (f64)(n * ITERATIONS * state.iterations()));
}

static inline std::vector<SortPattern> test_patterns() {
    return {
        SortPattern::unique_values,
        SortPattern::shuffled_16_values,
        SortPattern::all_equal,
    };
};

template <vector_machine M, i32 U, typename T>
void register_type(i64 s, SortPattern p) {
    if constexpr (U >= 2) {
        register_type<M, U / 2, T>(s, p);
    }
#ifdef VXSORT_COMPILER_MSVC
    auto realname = typeid(T).name();
#else
    auto realname = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
#endif
    auto bench_name = fmt::format("BM_vxsort_pattern<{}, {}, {}>/{}/{}", realname, U, s,
                                  magic_enum::enum_name(M), magic_enum::enum_name(p));
    ::benchmark::RegisterBenchmark(bench_name.c_str(), BM_vxsort_pattern<T, M, U>, s, p)
        ->Unit(kMillisecond)
        ->ThreadRange(1, processor_count);
}

template <vector_machine M, i32 U, typename... Ts>
void register_fullsort_benchmarks() {
    for (auto s : ::benchmark::CreateRange(MIN_SORT, MAX_SORT, 2)) {
        for (auto p : test_patterns()) {
            (register_type<M, U, Ts>(s, p), ...);
        }
    }
}

}  // namespace vxsort_bench

#endif  // VXSORT_BM_FULLSORT_VXSORT_H
