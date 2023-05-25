#include <random>
#include <algorithm>
#include <thread>
#include <benchmark/benchmark.h>

#include "../bench_isa.h"
#include <smallsort/bitonic_sort.h>

#include "../stolen-cycleclock.h"
#include "../util.h"

namespace vxsort_bench {
using namespace vxsort::types;
using vxsort::vector_machine;

const auto processor_count = std::thread::hardware_concurrency();

template <class Q, vxsort::vector_machine M>
static void BM_bitonic_sort(benchmark::State& state) {
    VXSORT_BENCH_ISA();

    using BM = vxsort::smallsort::bitonic<Q, M>;

    static const i32 ITERATIONS = 1024;
    auto n = state.range(0);
    auto v = generate_unique_values_vec(n, (Q)0x1000, (Q)0x8);

    auto copies = generate_copies(ITERATIONS, n, v);
    auto begins = generate_array_beginnings(copies);

    uint64_t total_cycles = 0;
    for (auto _ : state) {
        state.PauseTiming();
        refresh_copies(copies, v);
        state.ResumeTiming();
        auto start = cycleclock::Now();
        for (auto i = 0; i < ITERATIONS; i++)
            BM::sort(begins[i], n);
        total_cycles += cycleclock::Now() - start;
    }

    state.SetBytesProcessed(state.iterations() * n * ITERATIONS * sizeof(Q));

    state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
    process_perf_counters(state.counters, n * ITERATIONS);
    if (!state.counters.contains("cycles/N"))
        state.counters["rdtsc-cycles/N"] = make_cycle_per_n_counter((f64)total_cycles / (f64)(n * ITERATIONS * state.iterations()));
}

template <class Q, vxsort::vector_machine M, int N>
static void BM_bitonic_machine(benchmark::State& state) {
    VXSORT_BENCH_ISA();

    static_assert(N > 0, "N must be greater than 0");
    static_assert(N <= 4, "N cannot exceet 4");

    using BM = vxsort::smallsort::bitonic_machine<Q, M>;

    static const i32 ITERATIONS = 1024;
    auto n = N * BM::N;
    auto v = generate_unique_values_vec(n, (Q)0x1000, (Q)0x8);

    auto copies = generate_copies(ITERATIONS, n, v);
    auto begins = generate_array_beginnings(copies);

    uint64_t total_cycles = 0;
    for (auto _ : state) {
        state.PauseTiming();
        refresh_copies(copies, v);
        state.ResumeTiming();
        auto start = cycleclock::Now();
        for (auto i = 0; i < ITERATIONS; i++) {
            if (N == 1)
                BM::sort_01v_full_ascending(begins[i]);
            else if (N == 2)
                BM::sort_02v_full_ascending(begins[i]);
            else if (N == 3)
                BM::sort_03v_full_ascending(begins[i]);
            else if (N == 4)
                BM::sort_04v_full_ascending(begins[i]);

        }
        total_cycles += cycleclock::Now() - start;
    }

    state.SetBytesProcessed(state.iterations() * n * ITERATIONS * sizeof(Q));

    state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
    process_perf_counters(state.counters, n * ITERATIONS);
    if (!state.counters.contains("cycles/N"))
        state.counters["rdtsc-cycles/N"] = make_cycle_per_n_counter((f64)total_cycles / (f64)(n * ITERATIONS * state.iterations()));
}

}
