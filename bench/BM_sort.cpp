#include <random>
#include <algorithm>
#include <benchmark/benchmark.h>

#include <gcsort.h>

static void generate_unique_ptrs_vec(std::vector<uint8_t *>& vec, size_t n) {
    std::iota(vec.begin(), vec.end(), (uint8_t *) 0x1000);

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(vec.begin(), vec.end(), g);
}


static void BM_introsort(benchmark::State &state) {
    auto n = state.range(0);
    auto v = std::vector<uint8_t*>(n);
    auto begin = v.data();
    auto end = v.data() + v.size() - 1;

    for (auto _ : state) {
        state.PauseTiming();
        generate_unique_ptrs_vec(v, n);
        state.ResumeTiming();
        sort_introsort(begin, end);
    }

    state.counters["Time/N"] = benchmark::Counter(
            n,
            benchmark::Counter::Flags::kIsIterationInvariantRate | benchmark::Counter::Flags::kInvert,
            benchmark::Counter::kIs1000);
}

BENCHMARK(BM_introsort)->RangeMultiplier(2)->Range(4096, 1 << 18)->Unit(benchmark::kMillisecond);
