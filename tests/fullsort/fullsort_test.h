#ifndef VXSORT_FULLSORT_TEST_H
#define VXSORT_FULLSORT_TEST_H

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <vector>
#include <isa_detection.h>
#include <vxsort.h>


namespace vxsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using vxsort::vector_machine;


template <class T, int Unroll, vector_machine M>
void perform_vxsort_test(std::vector<T> V) {
  if (!vxsort::supports_vector_machine(M)) {
    GTEST_SKIP_(
        "Current CPU does not support the minimal features for this test");
    return;
  }
  auto begin = V.data();
  auto end = V.data() + V.size() - 1;

  auto sorter = vxsort::vxsort<T, M, Unroll>();
  sorter.sort(begin, end);

  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

}

#endif  // VXSORT_FULLSORT_TEST_H
