#include <random>
#include <algorithm>
#include <thread>
#include <benchmark/benchmark.h>

#include <isa_detection.h>
#include <smallsort/bitonic_sort.h>

#include "../stolen-cycleclock.h"
#include "../util.h"

namespace vxsort_bench {
using namespace vxsort::types;
using vxsort::vector_machine;

const auto processor_count = std::thread::hardware_concurrency();

template <class Q, vxsort::vector_machine M>
static void BM_bitonic_sort(benchmark::State& state) {
    if (!vxsort::supports_vector_machine(M)) {
        state.SkipWithError("Current CPU does not support the minimal features for this test");
        return;
    }

    static const i32 ITERATIONS = 1024;
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
