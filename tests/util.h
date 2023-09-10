#ifndef VXSORT_TEST_UTIL_H
#define VXSORT_TEST_UTIL_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include <defs.h>

namespace vxsort_tests {
using namespace vxsort::types;

const std::random_device::result_type global_bench_random_seed = 666;

template <typename IntType>
std::vector<IntType> range(IntType start, IntType stop, IntType step) {
    if (step == IntType(0)) {
        throw std::invalid_argument("step for range must be non-zero");
    }

    std::vector<IntType> result;
    IntType i = start;
    while ((step > 0) ? (i <= stop) : (i > stop)) {
        result.push_back(i);
        i += step;
    }

    return result;
}

template <typename IntType>
std::vector<IntType> multiply_range(IntType start, IntType stop, IntType step) {
    if (step == IntType(0)) {
        throw std::invalid_argument("step for range must be non-zero");
    }

    std::vector<IntType> result;
    IntType i = start;
    while ((step > 0) ? (i <= stop) : (i > stop)) {
        result.push_back(i);
        i *= step;
    }

    return result;
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
        v.push_back(start + stride * (i % 16));
    std::mt19937_64 rng(global_bench_random_seed);
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

template <typename T>
std::vector<T> all_equal(usize size, T start , T stride) {
    std::vector<T> v(size);
    for (i32 i = 0; i < size; ++i)
        v.push_back(start);
    return v;
}

template <typename T>
std::vector<T> ascending_int(usize size, T start, T stride) {
    std::vector<T> v(size);
    for (usize i = 0; i < size; ++i)
        v.push_back(start + stride * i);
    return v;
}

template <typename T>
std::vector<T> descending_int(usize size, T start, T stride) {
    std::vector<T> v(size);
    for (isize i = size - 1; i >= 0; --i)
        v.push_back(start + stride * i);
    return v;
}

template <typename T>
std::vector<T> pipe_organ(usize size, T start, T stride) {
    std::vector<T> v(size);
    for (usize i = 0; i < size/2; ++i)
        v.push_back(start + stride * i);
    for (usize i = size/2; i < size; ++i)
        v.push_back(start + (size - i) * stride);
    return v;
}

template <typename T>
std::vector<T> push_front(usize size, T start, T stride) {
    std::vector<T> v(size);
    for (usize i = 1; i < size; ++i)
        v.push_back(start + stride * i);
    v.push_back(start);
    return v;
}

template <typename T>
std::vector<T> push_middle(usize size, T start, T stride) {
    std::vector<T> v(size);
    for (usize i = 0; i < size; ++i) {
        if (i != size/2)
            v.push_back(start + stride * i);
    }
    v.push_back(start + stride * (size/2));
    return v;
}

}

#endif
