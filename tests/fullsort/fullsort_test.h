#ifndef VXSORT_FULLSORT_TEST_H
#define VXSORT_FULLSORT_TEST_H

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include <fmt/format.h>

#include "../test_isa.h"
#include "vxsort.h"

namespace vxsort_tests {
using namespace vxsort::types;
using ::vxsort::vector_machine;

template <typename T, i32 Unroll, vector_machine M>
void vxsort_test(std::vector<T>& V) {
    VXSORT_TEST_ISA();

    auto v_copy = std::vector<T>(V);
    auto begin = V.data();
    auto end = V.data() + V.size() - 1;

    auto sorter = ::vxsort::vxsort<T, M, Unroll>();
    sorter.sort(begin, end);

    std::sort(v_copy.begin(), v_copy.end());
    usize size = v_copy.size();
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

}

#endif  // VXSORT_FULLSORT_TEST_H
