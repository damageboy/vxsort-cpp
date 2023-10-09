#ifndef VXSORT_SMALLSORT_TEST_H
#define VXSORT_SMALLSORT_TEST_H

#include <functional>
#include <magic_enum.hpp>

#include "../sort_fixtures.h"
#include "gtest/gtest.h"

#include "../test_isa.h"
#include "fmt/format.h"
#include "smallsort/bitonic_sort.h"

namespace vxsort_tests {

using vxsort::vector_machine;

template <class T, vector_machine M>
void bitonic_machine_sort_pattern_test(sort_pattern pattern, usize size, T first_value, T stride) {
    VXSORT_TEST_ISA();

    using BM = vxsort::smallsort::bitonic_machine<T, M>;

    auto V = generate_values_by_pattern<T>(pattern, size, first_value, stride);

    auto v_copy = std::vector<T>(V);
    auto begin = V.data();

    BM::sort_full_vectors_ascending(begin, size);

    std::sort(v_copy.begin(), v_copy.end());
    for (usize i = 0; i < size; ++i) {
        if (v_copy[i] != V[i]) {
            GTEST_FAIL() << fmt::format("value at idx #{}  {} != {}", i, v_copy[i], V[i]);
        }
    }
}

template <class T, vector_machine M>
void bitonic_sort_pattern_test(sort_pattern pattern, usize size, T first_value, T stride) {
    VXSORT_TEST_ISA();

    auto V = generate_values_by_pattern<T>(pattern, size, first_value, stride);

    auto v_copy = std::vector<T>(V);
    auto begin = V.data();

    vxsort::smallsort::bitonic<T, M>::sort(begin, size);
    std::sort(v_copy.begin(), v_copy.end());
    for (usize i = 0; i < size; ++i) {
        if (v_copy[i] != V[i]) {
            GTEST_FAIL() << fmt::format("value at idx #{}  {} != {}", i, v_copy[i], V[i]);
        }
    }
}

static inline std::vector<sort_pattern> smallsort_test_patterns() {
    return {
        sort_pattern::unique_values,
        sort_pattern::shuffled_16_values,
        sort_pattern::all_equal,
    };
}

template <typename T>
struct smallsort_test_params {
public:
    smallsort_test_params(sort_pattern pattern, usize size, T first_value, T value_stride)
        : pattern(pattern), size(size), first_value(first_value), stride(value_stride) {}
    sort_pattern pattern;
    usize size;
    T first_value;
    T stride;
};

template <typename T>
std::vector<smallsort_test_params<T>> param_range(usize start,
                                                  usize stop,
                                                  usize step,
                                                  T first_value,
                                                  T value_stride) {
    assert(step > 0);

    auto patterns = smallsort_test_patterns();

    using TestParams = smallsort_test_params<T>;
    std::vector<TestParams> tests;

    for (const auto& p : smallsort_test_patterns()) {
        for (usize i = start; i <= stop; i += step) {
            if (static_cast<i64>(i) <= 0)
                continue;

            tests.push_back(TestParams(p, i, first_value, value_stride));
        }
    }
    return tests;
}

template <vector_machine M, typename T>
void register_bitonic_tests(usize test_size_bytes, T first_value, T value_stride) {
    auto stop = test_size_bytes / sizeof(T);
    usize step = 1;
    auto tests = param_range(1, stop, step, first_value, value_stride);

    for (auto p : tests) {
        auto* test_type = get_canonical_typename<T>();

        auto test_size = p.size;
        auto test_name =
            fmt::format("bitonic_sort_pattern_test<{}, {}>/{}/{}", test_type,
                        magic_enum::enum_name(M), magic_enum::enum_name(p.pattern), test_size);

        register_single_test_lambda("smallsort", test_name.c_str(), nullptr,
                                    std::to_string(test_size).c_str(),
                                    __FILE__, __LINE__,
                                    bitonic_sort_pattern_test<T, M>, p.pattern, test_size,
                                    p.first_value, p.stride);
    }
}

template <vector_machine M, typename T>
void register_bitonic_machine_tests(T first_value, T value_stride) {
    using VM = vxsort::vxsort_machine_traits<T, M>;

    // We test bitonic_machine from 1 up to 4 vectors in single vector increments
    //auto stop = (sizeof(typename VM::TV) * 4) / sizeof(T);
    auto stop = (sizeof(typename VM::TV) * 1) / sizeof(T);
    usize step = sizeof(typename VM::TV) / sizeof(T);
    assert(step > 0);

    auto tests = param_range(step, stop, step, first_value, value_stride);

    for (auto p : tests) {
        auto* test_type = get_canonical_typename<T>();

        auto test_size = p.size;
        auto test_name =
            fmt::format("bitonic_machine_sort_pattern_test<{}, {}>/{}/{}", test_type,
                        magic_enum::enum_name(M), magic_enum::enum_name(p.pattern), test_size);

        register_single_test_lambda("smallsort", test_name.c_str(), nullptr,
                                    std::to_string(test_size).c_str(),
                                    __FILE__, __LINE__,
                                    bitonic_machine_sort_pattern_test<T, M>, p.pattern, test_size,
                                    p.first_value, p.stride);
    }
}
}  // namespace vxsort_tests

#endif  // VXSORT_SMALLSORT_TEST_H
