#ifndef GCSORT_FULLSORT_H
#define GCSORT_FULLSORT_H

#include "gmock/gmock.h"
#include "gtest/gtest.h"


#include <vector>
#include <vxsort.h>

namespace gcsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using gcsort::vector_machine;


template <class T, int Unroll, vector_machine M>
void perform_vxsort_test(std::vector<T> V) {
  auto begin = V.data();
  auto end = V.data() + V.size() - 1;

  auto sorter = gcsort::vxsort<T, M, Unroll>();
  sorter.sort(begin, end);

  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

}

#endif  // GCSORT_FULLSORT_H
