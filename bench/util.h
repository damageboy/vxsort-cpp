#ifndef VXSORT_BENCH_UTIL_H
#define VXSORT_BENCH_UTIL_H

#include <benchmark/benchmark.h>

#include <defs.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>
#ifndef VXSORT_COMPILER_MSVC
#include <cxxabi.h>
#endif


#include "stolen-cycleclock.h"

using namespace benchmark;

namespace vxsort_bench {

using namespace vxsort::types;

Counter make_time_per_n_counter(i64 n);

Counter make_cycle_per_n_counter(f64 n);

std::string get_crypto_hash(void *start, void *end);

void process_perf_counters(UserCounters &counters, i64 num_elements);

extern std::random_device::result_type global_bench_random_seed;

template <typename T>
void refresh_copies(std::vector<std::vector<T>> &copies, std::vector<T>& orig) {
    const auto begin = orig.begin();
    const auto end = orig.end();
    const auto num_copies = copies.size();
    for (usize i = 0; i < num_copies; i++)
        copies[i].assign(begin, end);
}

template <typename T>
std::vector<std::vector<T>> generate_copies(usize num_copies, i64 n, std::vector<T>& orig) {
    std::vector<std::vector<T>> copies(num_copies);
    for (usize i = 0; i < num_copies; i++)
        copies[i] = std::vector<T>(n);
    refresh_copies(copies, orig);
    return copies;
}

template <typename T, typename U=T>
std::vector<U *> generate_array_beginnings(std::vector<std::vector<T>> &copies) {
    const auto num_copies = copies.size();
    std::vector<U*> begins(num_copies);
    for (usize i = 0; i < num_copies; i++)
        begins[i] = (U*)copies[i].data();
    return begins;
}

template <typename T>
std::vector<T> unique_values(usize size, T start, T stride) {
    std::vector<T> v(size);
    for (usize i = 0; i < v.size(); i++, start += stride)
        v[i] = start;

    std::mt19937_64 rng(global_bench_random_seed);
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

template <typename T>
std::vector<T> shuffled_16_values(usize size, T start, T stride) {
    std::vector<T> v(size);
    for (usize i = 0; i < size; ++i)
        v[i] = start + stride * (i % 16);

    std::mt19937_64 rng(global_bench_random_seed);
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

template <typename T>
std::vector<T> all_equal(usize size, T start , T) {
    std::vector<T> v(size);
    for (usize i = 0; i < size; ++i)
        v[i] = start;
    return v;
}

template <typename T>
std::vector<T> ascending_int(usize size, T start, T stride) {
    std::vector<T> v(size);
    for (usize i = 0; i < size; ++i)
        v[i] = start + stride * i;
    return v;
}

template <typename T>
std::vector<T> descending_int(usize size, T start, T stride) {
    std::vector<T> v(size);
    for (isize i = size - 1; i >= 0; --i)
        v[i] = start + stride * i;
    return v;
}

template <typename T>
std::vector<T> pipe_organ(usize size, T start, T stride) {
    std::vector<T> v(size);
    for (usize i = 0; i < size/2; ++i)
        v[i] = start + stride * i;
    for (usize i = size/2; i < size; ++i)
        v[i] = start + (size - i) * stride;
    return v;
}

template <typename T>
std::vector<T> push_front(usize size, T start, T stride) {
    std::vector<T> v(size);
    for (usize i = 1; i < size; ++i)
        v[i-1] = start + stride * i;
    v[size-1] = start;
    return v;
}

template <typename T>
std::vector<T> push_middle(usize size, T start, T stride) {
    std::vector<T> v(size);
    for (usize i = 0; i < size; ++i) {
        if (i != size/2)
            v[i] = start + stride * i;
    }
    v[size/2] = start + stride * (size/2);
    return v;
}

template<typename T>
const char *get_canonical_typename() {
#ifdef VXSORT_COMPILER_MSVC
    auto realname = typeid(T).name();
#else
    auto realname = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
#endif

    if (realname == nullptr) {
        return "unknown";
    } else if (std::strcmp(realname, "long") == 0)
        return "i64";
    else if (std::strcmp(realname, "unsigned long") == 0)
        return "u64";
    else if (std::strcmp(realname, "int") == 0)
        return "i32";
    else if (std::strcmp(realname, "unsigned int") == 0)
        return "u32";
    else if (std::strcmp(realname, "short") == 0)
        return "i16";
    else if (std::strcmp(realname, "unsigned short") == 0)
        return "u16";
    else if (std::strcmp(realname, "char") == 0)
        return "i8";
    else if (std::strcmp(realname, "unsigned char") == 0)
        return "u8";
    else if (std::strcmp(realname, "float") == 0)
        return "f32";
    else if (std::strcmp(realname, "double") == 0)
        return "f64";
    else
        return realname;
}


}

#endif //VXSORT_BENCH_UTIL_H
