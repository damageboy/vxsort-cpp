#include <random>
#include <algorithm>
#include <thread>
#include <benchmark/benchmark.h>

#include "../reference/introsort/introsort_orig.h"
#include <isa_detection.h>
#include <smallsort/bitonic_sort.h>

#include "../stolen-cycleclock.h"
#include "../util.h"

namespace vxsort_bench {
using namespace vxsort::types;
using vxsort::vector_machine;

const auto processor_count = std::thread::hardware_concurrency();

template <typename T>
static void insertionsort_cpp(T* lo, T* hi) {
    for (T* i = lo + 1; i <= hi; i++) {
        T* j = i;
        T t = *i;
        while ((j > lo) && (t < *(j - 1))) {
            *j = *(j - 1);
            j--;
        }
        *j = t;
    }
}

template <class Q>
static void BM_insertion_sort(benchmark::State& state) {
    static const int ITERATIONS = 1024;
    auto n = state.range(0);
    auto v = std::vector<Q>(n);
    generate_unique_values_vec(v, (Q)0x1000, (Q)8);

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
            insertionsort_cpp((Q*)begins[i], (Q*)ends[i]);
        total_cycles += cycleclock::Now() - start;
    }

    state.SetBytesProcessed(state.iterations() * n * ITERATIONS * sizeof(Q));

    state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
    process_perf_counters(state.counters, n * ITERATIONS);
    if (!state.counters.contains("cycles/N"))
        state.counters["rdtsc-cycles/N"] = make_cycle_per_n_counter((f64)total_cycles / (f64)(n * ITERATIONS * state.iterations()));
}

template <class Q, vxsort::vector_machine M>
static void BM_bitonic_sort(benchmark::State& state) {
    if (!vxsort::supports_vector_machine(M)) {
        state.SkipWithError("Current CPU does not support the minimal features for this test");
        return;
    }

    static const int ITERATIONS = 1024;
    auto n = state.range(0);
    auto v = std::vector<Q>(n);
    generate_unique_values_vec(v, (Q)0x1000, (Q)0x8);

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

    state.SetBytesProcessed(state.iterations() * n * ITERATIONS * sizeof(Q));

    state.counters["Time/N"] = make_time_per_n_counter(n * ITERATIONS);
    process_perf_counters(state.counters, n * ITERATIONS);
    if (!state.counters.contains("cycles/N"))
        state.counters["rdtsc-cycles/N"] = make_cycle_per_n_counter((f64)total_cycles / (f64)(n * ITERATIONS * state.iterations()));

}

}
