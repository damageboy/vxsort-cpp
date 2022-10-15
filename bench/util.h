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
void generate_unique_values_vec(std::vector<T>& vec, T start, T stride) {
    for (isize i = 0; i < vec.size(); i++, start += stride)
        vec[i] = start;

    std::random_device rd;
    std::mt19937_64 g(rd());

    std::shuffle(vec.begin(), vec.end(), g);
}

template <typename T, typename U=T>
std::vector<U *> generate_array_beginnings(std::vector<std::vector<T>> &copies) {
    const auto num_copies = copies.size();
    std::vector<U*> begins(num_copies);
    for (isize i = 0; i < num_copies; i++)
        begins[i] = (U*)copies[i].data();
    return begins;
}

template <typename T>
void refresh_copies(std::vector<std::vector<T>> &copies, std::vector<T>& orig) {
    const auto begin = orig.begin();
    const auto end = orig.end();
    const auto num_copies = copies.size();
    for (isize i = 0; i < num_copies; i++)
        copies[i].assign(begin, end);
}

template <typename T>
std::vector<std::vector<T>> generate_copies(isize num_copies, i64 n, std::vector<T>& orig) {
    std::vector<std::vector<T>> copies(num_copies);
    for (isize i = 0; i < num_copies; i++)
        copies[i] = std::vector<T>(n);
    refresh_copies(copies, orig);
    return copies;
}

template <typename T>
std::vector<T> shuffled_seq(isize size, T start, T stride, std::mt19937_64& rng) {
    std::vector<T> v; v.reserve(size);
    for (isize i = 0; i < size; ++i)
        v.push_back(start + stride * i);
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

template <typename T>
std::vector<int> shuffled_16_values(isize size, T start, T stride, std::mt19937_64& rng) {
    std::vector<T> v; v.reserve(size);
    for (i32 i = 0; i < size; ++i)
        v.push_back(start + stride * (i % 16));
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

template <typename T>
std::vector<int> all_equal(isize size, T start) {
    std::vector<T> v; v.reserve(size);
    for (i32 i = 0; i < size; ++i)
        v.push_back(start);
    return v;
}

template <typename T>
std::vector<T> ascending_int(isize size, T start, T stride) {
    std::vector<T> v; v.reserve(size);
    for (isize i = 0; i < size; ++i)
        v.push_back(start + stride * i);
    return v;
}

template <typename T>
std::vector<T> descending_int(isize size, T start, T stride) {
    std::vector<T> v; v.reserve(size);
    for (isize i = size - 1; i >= 0; --i)
        v.push_back(start + stride * i);
    return v;
}

template <typename T>
std::vector<int> pipe_organ(isize size, T start, T stride, std::mt19937_64&) {
    std::vector<T> v; v.reserve(size);
    for (isize i = 0; i < size/2; ++i)
        v.push_back(start + stride * i);
    for (isize i = size/2; i < size; ++i)
        v.push_back(start + (size - i) * stride);
    return v;
}

template <typename T>
std::vector<int> push_front(isize size, T start, T stride, std::mt19937_64&) {
    std::vector<int> v; v.reserve(size);
    for (isize i = 1; i < size; ++i)
        v.push_back(start + stride * i);
    v.push_back(start);
    return v;
}

template <typename T>
std::vector<T> push_middle(isize size, T start, T stride, std::mt19937_64&) {
    std::vector<T> v; v.reserve(size);
    for (isize i = 0; i < size; ++i) {
        if (i != size/2)
            v.push_back(start + stride * i);
    }
    v.push_back(start + stride * (size/2));
    return v;
}

}

#endif //VXSORT_BENCH_UTIL_H
