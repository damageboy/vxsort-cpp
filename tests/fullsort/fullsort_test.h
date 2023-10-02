#ifndef VXSORT_FULLSORT_TEST_H
#define VXSORT_FULLSORT_TEST_H

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include <fmt/format.h>
#include <magic_enum.hpp>

#include "../test_vectors.h"
#include "../sort_fixtures.h"
#include "../test_isa.h"
#include "vxsort.h"

namespace vxsort_tests {
using namespace vxsort::types;
using ::vxsort::vector_machine;

template <typename T, i32 Unroll, vector_machine M>
void vxsort_pattern_test(sort_pattern, usize size, T first_value, T stride) {
    VXSORT_TEST_ISA();

    auto V = unique_values<T>(size, first_value, stride);

    auto v_copy = std::vector<T>(V);
    auto begin = V.data();
    auto end = V.data() + V.size() - 1;

    auto sorter = ::vxsort::vxsort<T, M, Unroll>();
    sorter.sort(begin, end);

    std::sort(v_copy.begin(), v_copy.end());
    for (usize i = 0; i < size; ++i) {
        if (v_copy[i] != V[i]) {
            GTEST_FAIL() << fmt::format("value at idx #{}  {} != {}", i, v_copy[i], V[i]);
        }
    }
}

template <typename T, i32 Unroll, int Shift, vector_machine M>
void vxsort_hinted_test(std::vector<T>& V, T min_value, T max_value) {
    VXSORT_TEST_ISA();

    auto v_copy = std::vector<T>(V);
    auto begin = V.data();
    auto end = V.data() + V.size() - 1;

    auto sorter = ::vxsort::vxsort<T, M, Unroll, Shift>();
    sorter.sort(begin, end, min_value, max_value);

    std::sort(v_copy.begin(), v_copy.end());
    usize size = v_copy.size();
    for (usize i = 0; i < size; ++i) {
        if (v_copy[i] != V[i]) {
            GTEST_FAIL() << fmt::format("value at idx #{}  {} != {}", i, v_copy[i], V[i]);
        }
    }
}

static inline std::vector<sort_pattern> fullsort_test_patterns() {
    return {
            sort_pattern::unique_values,
            //sort_pattern::shuffled_16_values,
            //sort_pattern::all_equal,
    };
}

template <typename T>
struct fullsort_test_params {
public:
    fullsort_test_params(sort_pattern pattern, usize size, i32 slack, T first_value, T value_stride)
            : pattern(pattern), size(size), slack(slack), first_value(first_value), stride(value_stride) {}
    sort_pattern pattern;
    usize size;
    i32 slack;
    T first_value;
    T stride;
};

template<typename T>
std::vector<fullsort_test_params<T>>
gen_params(usize start, usize stop, usize step, i32 slack, T first_value, T value_stride)
{
    auto patterns = fullsort_test_patterns();

    using TestParams = fullsort_test_params<T>;
    std::vector<TestParams> tests;

    for (auto p : fullsort_test_patterns()) {
        for (auto i : multiply_range<i32>(start, stop, step)) {
            for (auto j : range<i32>(-slack, slack, 1)) {
                if ((i64)i + j <= 0)
                    continue;
                tests.push_back(fullsort_test_params<T>(p, i, j, first_value, value_stride));
            }
        }
    }
    return tests;
}

template <vector_machine M, i32 U, typename T>
void register_fullsort_tests(usize start, usize stop, usize step, T first_value, T value_stride) {
    if (step == 0) {
        throw std::invalid_argument("step for range must be non-zero");
    }

    if constexpr (U >= 2) {
        register_fullsort_tests<M, U / 2, T>(start, stop, step, first_value, value_stride);
    }

    using VM = vxsort::vxsort_machine_traits<T, M>;

    // Test "slacks" are defined in terms of number of elements in the primitive size (T)
    // up to the number of such elements contained in one vector type (VM::TV)
    constexpr i32 slack = sizeof(typename VM::TV) /  sizeof(T);
    static_assert(slack > 1);

    auto tests = gen_params(start, stop, step, slack, first_value, value_stride);

    for (auto p : tests) {
        auto *test_type = get_canonical_typename<T>();

        auto test_size = p.size + p.slack;
        auto test_name = fmt::format("vxsort_pattern_test<{}, {}, {}>/{}/{}", test_type, U,
                                     magic_enum::enum_name(M), magic_enum::enum_name(p.pattern), test_size);

        RegisterSingleLambdaTest(
                "fullsort", test_name.c_str(), nullptr,
                std::to_string(test_size).c_str(),
                __FILE__, __LINE__,
                vxsort_pattern_test<T, U, M>, p.pattern, test_size, p.first_value, p.stride);
    }
}
}

#endif  // VXSORT_FULLSORT_TEST_H
