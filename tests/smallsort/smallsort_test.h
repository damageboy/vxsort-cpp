#ifndef VXSORT_SMALLSORT_TEST_H
#define VXSORT_SMALLSORT_TEST_H

#include <functional>

#include "gtest/gtest.h"
#include "../sort_fixtures.h"

#include "../test_isa.h"
#include "smallsort/bitonic_sort.h"
#include "fmt/format.h"

namespace vxsort_tests {

using vxsort::vector_machine;

template <class T, vector_machine M, bool ascending>
void bitonic_machine_sort_test(std::vector<T>& V) {
    VXSORT_TEST_ISA();

    using BM = vxsort::smallsort::bitonic_machine<T, M>;

    auto v_copy = std::vector<T>(V);
    auto begin = V.data();
    auto size = V.size();

    if (ascending)
        BM::sort_full_vectors_ascending(begin, size);
    else
        BM::sort_full_vectors_descending(begin, size);

    std::sort(v_copy.begin(), v_copy.end());
    for (usize i = 0; i < size; ++i) {
        if (v_copy[i] != V[i]) {
            GTEST_FAIL() << fmt::format("value at idx #{}  {} != {}", i, v_copy[i], V[i]);
        }
    }
}

template <class T, vector_machine M>
void bitonic_sort_test(std::vector<T>& V) {
    VXSORT_TEST_ISA();

    auto v_copy = std::vector<T>(V);
    auto begin = V.data();
    auto size = V.size();

    vxsort::smallsort::bitonic<T, M>::sort(begin, size);
    std::sort(v_copy.begin(), v_copy.end());
    for (usize i = 0; i < size; ++i) {
        if (v_copy[i] != V[i]) {
            GTEST_FAIL() << fmt::format("value at idx #{}  {} != {}", i, v_copy[i], V[i]);
        }
    }
}
}

#endif  // VXSORT_SMALLSORT_TEST_H
