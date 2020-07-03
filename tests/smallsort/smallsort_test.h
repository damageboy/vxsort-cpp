#ifndef VXSORT_SMALLSORT_TEST_H
#define VXSORT_SMALLSORT_TEST_H

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "../fixtures.h"

#include "isa_detection.h"
#include "smallsort/bitonic_sort.h"


namespace vxsort_tests {

using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using vxsort::vector_machine;

template <class T, vector_machine M>
void perform_bitonic_sort_test(std::vector<T> V) {

  if (!vxsort::supports_vector_machine(M)) {
    GTEST_SKIP_(
        "Current CPU does not support the minimal features for this test");
    return;
  }
  auto begin = V.data();
  vxsort::smallsort::bitonic<T, M>::sort(begin, V.size());
  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

template <class T, vector_machine M>
void perform_bitonic_alt_sort_test(std::vector<T> V) {

    if (!vxsort::supports_vector_machine(M)) {
        GTEST_SKIP_(
            "Current CPU does not support the minimal features for this test");
        return;
    }
    auto begin = V.data();
    vxsort::smallsort::bitonic<T, M>::sort_alt(begin, V.size());
    EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}


}

#endif  // VXSORT_SMALLSORT_TEST_H
