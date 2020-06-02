#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "../fixtures.h"

#include <smallsort/bitonic_sort.AVX2.double.generated.h>
#include <smallsort/bitonic_sort.AVX512.double.generated.h>

#include <algorithm>
#include <iterator>

namespace gcsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using gcsort::vector_machine;

struct BitonicAVX2_double : public SortTest<double> {};
struct BitonicAVX512_double : public SortTest<double> {};


INSTANTIATE_TEST_SUITE_P(BitonicSizes,
                         BitonicAVX2_double,
                         ValuesIn(range(4, 64, 4)),
                         PrintValue());

TEST_P(BitonicAVX2_double, BitonicSort) {
  auto begin = V.data();
  gcsort::smallsort::bitonic<double, vector_machine::AVX2>::sort(begin, GetParam());
  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}


INSTANTIATE_TEST_SUITE_P(Bitonic,
                         BitonicAVX512_double,
                         ValuesIn(range(8, 128, 8)),
                         PrintValue());

TEST_P(BitonicAVX512_double, BitonicSortAVX512) {
  auto begin = V.data();
  gcsort::smallsort::bitonic<double, gcsort::AVX512>::sort(begin, GetParam());
  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

}