#ifndef VXSORT_BENCH_UTIL_H
#define VXSORT_BENCH_UTIL_H

#include <benchmark/benchmark.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include <defs.h>

using namespace benchmark;

namespace vxsort_bench {

using namespace vxsort::types;

Counter make_time_per_n_counter(i64 n);

Counter make_cycle_per_n_counter(f64 n);

void process_perf_counters(UserCounters &counters, i64 num_elements);

template <typename T>
extern void generate_unique_values_vec(std::vector<T>& vec, T start, T stride) {
    for (usize i = 0; i < vec.size(); i++, start += stride)
        vec[i] = start;

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(vec.begin(), vec.end(), g);
}

template <typename T, typename U=T>
extern std::vector<U *> generate_array_beginnings(std::vector<std::vector<T>> &copies) {
    const auto num_copies = copies.size();
    std::vector<U*> begins(num_copies);
    for (usize i = 0; i < num_copies; i++)
        begins[i] = (U*)copies[i].data();
    return begins;
}
template <typename T>
extern void refresh_copies(std::vector<std::vector<T>> &copies, std::vector<T>& orig) {
    const auto begin = orig.begin();
    const auto end = orig.end();
    const auto num_copies = copies.size();
    for (usize i = 0; i < num_copies; i++)
        copies[i].assign(begin, end);
}

template <typename T>
extern std::vector<std::vector<T>> generate_copies(usize num_copies,  i64 n, std::vector<T>& orig) {
    std::vector<std::vector<T>> copies(num_copies);
    for (usize i = 0; i < num_copies; i++)
        copies[i] = std::vector<T>(n);
    refresh_copies(copies, orig);
    return copies;
}

}

#endif //VXSORT_BENCH_UTIL_H
