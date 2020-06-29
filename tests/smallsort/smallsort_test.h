#ifndef VXSORT_SMALLSORT_TEST_H
#define VXSORT_SMALLSORT_TEST_H

#include <functional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "../fixtures.h"


#include "isa_detection.h"
#include "smallsort/bitonic_sort.h"

namespace vxsort_tests {

using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::WhenSortedBy;
using testing::Types;

using vxsort::vector_machine;

template <class T, vector_machine M, bool ascending>
void bitonic_machine_sort_test(std::vector<T> V) {
    if (!vxsort::supports_vector_machine(M)) {
        GTEST_SKIP_("Current CPU does not support the minimal features for this test");
        return;
    }
    auto begin = V.data();
    auto size = V.size();
    auto v_copy = V;

    if (ascending)
        vxsort::smallsort::bitonic_machine<T, M>::sort_full_vectors_ascending(begin, size);
    else
        vxsort::smallsort::bitonic_machine<T, M>::sort_full_vectors_descending(begin, size);

    std::sort(v_copy.begin(), v_copy.end());

    EXPECT_THAT(V, ElementsAreArray(v_copy));
//

//    if (ascending)
//        EXPECT_THAT(V, WhenSortedBy(std::less<T>(), ElementsAreArray(V)));
//    else
//        EXPECT_THAT(V, WhenSortedBy(std::greater<T>(), ElementsAreArray(V)));
}

template <class T, vector_machine M>
void bitonic_sort_test(std::vector<T> V) {
    if (!vxsort::supports_vector_machine(M)) {
        GTEST_SKIP_("Current CPU does not support the minimal features for this test");
        return;
    }
    auto begin = V.data();
    auto size = V.size();
    {
        auto v_copy = V;

        vxsort::smallsort::bitonic<T, M>::sort(begin, size);

        std::sort(v_copy.begin(), v_copy.end());

        EXPECT_THAT(V, ElementsAreArray(v_copy));
    }
}



}

#endif  // VXSORT_SMALLSORT_TEST_H
