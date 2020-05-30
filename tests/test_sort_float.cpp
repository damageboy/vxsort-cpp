#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fixtures.h"

#include <vxsort.h>
#include <smallsort/bitonic_sort.h>
#include <smallsort/bitonic_sort.AVX2.float.generated.h>

#include <algorithm>
#include <iterator>

namespace gcsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using gcsort::vxsort;
using gcsort::vector_machine;

struct SmallSortTest_float : public SortTest<float> {};


INSTANTIATE_TEST_SUITE_P(BitonicSizes,
                         SmallSortTest_float,
                         ValuesIn(range(8, 128, 8)),
                         PrintValue());

TEST_P(SmallSortTest_float, BitonicSort) {
  auto begin = V.data();
  gcsort::smallsort::bitonic<float>::sort(begin, GetParam());
  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}


struct FullSortTest_float : public SortWithSlackTest<float> {};


INSTANTIATE_TEST_SUITE_P(FullSortSizes,
                         FullSortTest_float,
                         ValuesIn(SizeAndSlack::generate(10, 1000000, 10, FullSortTest_float::VectorElements*2)),
                         PrintSizeAndSlack());


TEST_P(FullSortTest_float, VxSort) {
  auto begin = V.data();
  auto end = V.data() + V.size() - 1;

  auto sorter = gcsort::vxsort<float, vector_machine::AVX2, 1>();
  sorter.sort(begin, end);

  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

TEST_P(FullSortTest_float, VxSort_Unroll2) {
  auto begin = V.data();
  auto end = V.data() + V.size() - 1;

  auto sorter = gcsort::vxsort<float, vector_machine::AVX2, 2>();
  sorter.sort(begin, end);

  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

TEST_P(FullSortTest_float, VxSort_Unroll4) {
  auto begin = V.data();
  auto end = V.data() + V.size() - 1;

  auto sorter = gcsort::vxsort<float, vector_machine::AVX2, 4>();
  sorter.sort(begin, end);

  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

TEST_P(FullSortTest_float, VxSort_Unroll8) {
  auto begin = V.data();
  auto end = V.data() + V.size() - 1;

  auto sorter = gcsort::vxsort<float, vector_machine::AVX2, 8>();
  sorter.sort(begin, end);

  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}


TEST_P(FullSortTest_float, VxSort_Unroll12) {
  auto begin = V.data();
  auto end = V.data() + V.size() - 1;

  auto sorter = gcsort::vxsort<float, vector_machine::AVX2, 12>();
  sorter.sort(begin, end);

  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}


}