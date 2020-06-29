#ifndef VXSORT_FULLSORT_TEST_H
#define VXSORT_FULLSORT_TEST_H

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <fmt/format.h>

#include "isa_detection.h"
#include "vxsort.h"

namespace vxsort_tests {
using namespace vxsort::types;
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using ::vxsort::vector_machine;

template <typename T, int Unroll, vector_machine M>
void vxsort_test(std::vector<T> V) {
    if (!::vxsort::supports_vector_machine(M)) {
        GTEST_SKIP_("Current CPU does not support the minimal features for this test");
        return;
    }

    auto copy = std::vector<T>(V);

    auto begin = V.data();
    auto end = V.data() + V.size() - 1;

    auto sorter = ::vxsort::vxsort<T, M, Unroll>();
    sorter.sort(begin, end);

    //EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
    std::sort(copy.begin(), copy.end());
    u64 size = copy.size();
    for (auto i = 0ULL; i < size; ++i) {
        if (copy[i] != V[i]) {
            GTEST_FAIL() << fmt::format("value at idx #{}  {} != {}", i, copy[i], V[i]);
        }
    }

}

template <typename T, int Unroll, int Shift, vector_machine M>
void vxsort_hinted_test(std::vector<T> V, T min_value, T max_value) {
    if (!::vxsort::supports_vector_machine(M)) {
        GTEST_SKIP_(
                "Current CPU does not support the minimal features for this test");
        return;
    }

    auto copy = std::vector<T>(V);

    auto begin = V.data();
    auto end = V.data() + V.size() - 1;

    auto sorter = ::vxsort::vxsort<T, M, Unroll, Shift>();
    sorter.sort(begin, end, min_value, max_value);

    std::sort(copy.begin(), copy.end());

    //EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
    u64 size = copy.size();
    for (auto i = 0ULL; i < size; ++i) {
        if (copy[i] != V[i]) {
            GTEST_FAIL() << fmt::format("value at idx #{}  {} != {}", i, copy[i], V[i]);
        }
    }

}

}

#endif  // VXSORT_FULLSORT_TEST_H
