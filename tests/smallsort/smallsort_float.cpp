#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "../fixtures.h"

#include <smallsort/bitonic_sort.AVX2.float.generated.h>
#include <smallsort/bitonic_sort.AVX512.float.generated.h>

#include <algorithm>
#include <iterator>

namespace gcsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using gcsort::vector_machine;

struct BitonicAVX2_float : public SortTest<float> {};

struct BitonicAVX512_float : public SortTest<float> {};


INSTANTIATE_TEST_SUITE_P(Bitonic,
                         BitonicAVX2_float,
                         ValuesIn(range(8, 128, 8)),
                         PrintValue());

TEST_P(BitonicAVX2_float, BitonicSortAVX2) {
  auto begin = V.data();
  gcsort::smallsort::bitonic<float, vector_machine::AVX2>::sort(begin, GetParam());
  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}


INSTANTIATE_TEST_SUITE_P(Bitonic,
                         BitonicAVX512_float,
                         ValuesIn(range(16, 256, 16)),
                         PrintValue());

TEST_P(BitonicAVX512_float, BitonicSortAVX512) {
  auto begin = V.data();
  gcsort::smallsort::bitonic<float, vector_machine::AVX512>::sort(begin, GetParam());
  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

}