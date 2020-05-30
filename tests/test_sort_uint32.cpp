#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fixtures.h"

#include <vxsort.h>
#include <smallsort/bitonic_sort.h>
#include <smallsort/bitonic_sort.AVX2.int32_t.generated.h>

#include <algorithm>
#include <iterator>

namespace gcsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using gcsort::vxsort;
using gcsort::vector_machine;

struct SmallSortTest_ui32 : public SortTest<uint32_t> {};

INSTANTIATE_TEST_SUITE_P(BitonicSizes,
                         SmallSortTest_ui32,
                         ValuesIn(range(8, 128, 8)),
                         PrintValue());

TEST_P(SmallSortTest_ui32, BitonicSort) {
    auto begin = V.data();
    gcsort::smallsort::bitonic<uint32_t>::sort(begin, GetParam());
    EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}


struct FullSortTest_ui32 : public SortWithSlackTest<uint32_t> {};

INSTANTIATE_TEST_SUITE_P(FullSortSizes,
                         FullSortTest_ui32,
                         ValuesIn(SizeAndSlack::generate(10, 1000000, 10, FullSortTest_ui32::VectorElements*2)),
                         PrintSizeAndSlack());


TEST_P(FullSortTest_ui32, VxSort) {
    auto begin = V.data();
    auto end = V.data() + V.size() - 1;

    auto sorter = gcsort::vxsort<uint32_t, vector_machine::AVX2, 1>();
    sorter.sort(begin, end);

    EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

TEST_P(FullSortTest_ui32, VxSort_Unroll2) {
  auto begin = V.data();
  auto end = V.data() + V.size() - 1;

  auto sorter = gcsort::vxsort<uint32_t, vector_machine::AVX2, 2>();
  sorter.sort(begin, end);

  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

TEST_P(FullSortTest_ui32, VxSort_Unroll4) {
  auto begin = V.data();
  auto end = V.data() + V.size() - 1;

  auto sorter = gcsort::vxsort<uint32_t, vector_machine::AVX2, 4>();
  sorter.sort(begin, end);

  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

TEST_P(FullSortTest_ui32, VxSort_Unroll8) {
  auto begin = V.data();
  auto end = V.data() + V.size() - 1;

  auto sorter = gcsort::vxsort<uint32_t, vector_machine::AVX2, 8>();
  sorter.sort(begin, end);

  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

TEST_P(FullSortTest_ui32, VxSort_Unroll12) {
  auto begin = V.data();
  auto end = V.data() + V.size() - 1;

  auto sorter = gcsort::vxsort<uint32_t, vector_machine::AVX2, 12>();
  sorter.sort(begin, end);

  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}
}