#ifndef VXSORT_TEST_UTIL_H
#define VXSORT_TEST_UTIL_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#ifndef VXSORT_COMPILER_MSVC
#include <cxxabi.h>
#endif
#include <cstring>
#include <defs.h>

namespace vxsort_tests {
using namespace vxsort::types;

enum class sort_pattern {
    unique_values,
    shuffled_16_values,
    all_equal,
    ascending_int,
    descending_int,
    pipe_organ,
    push_front,
    push_middle
};

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

template <typename T>
std::vector<T>
generate_values_by_pattern(sort_pattern pattern, usize size, T first_value, T stride)
{
    switch (pattern) {
        case sort_pattern::unique_values:
            return unique_values<T>(size, first_value, stride);
        case sort_pattern::shuffled_16_values:
            return shuffled_16_values<T>(size, first_value, stride);
        case sort_pattern::all_equal:
            return all_equal<T>(size, first_value, stride);
        case sort_pattern::ascending_int:
            return ascending_int<T>(size, first_value, stride);
        case sort_pattern::descending_int:
            return descending_int<T>(size, first_value, stride);
        case sort_pattern::pipe_organ:
            return pipe_organ<T>(size, first_value, stride);
        case sort_pattern::push_front:
            return push_front<T>(size, first_value, stride);
        case sort_pattern::push_middle:
            return push_middle<T>(size, first_value, stride);
        default:
            throw std::invalid_argument("unknown sort pattern");
    }

}

}

#endif
