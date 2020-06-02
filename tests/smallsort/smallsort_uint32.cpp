#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "../fixtures.h"

#include <smallsort/bitonic_sort.AVX2.uint32_t.generated.h>
#include <smallsort/bitonic_sort.AVX512.uint32_t.generated.h>

#include <algorithm>
#include <iterator>

namespace gcsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using gcsort::vector_machine;

struct BitonicAVX2_ui32 : public SortTest<uint32_t> {};

struct BitonicAVX512_ui32 : public SortTest<uint32_t> {};

INSTANTIATE_TEST_SUITE_P(Bitonic,
                         BitonicAVX2_ui32,
                         ValuesIn(range(8, 128, 8)),
                         PrintValue());

TEST_P(BitonicAVX2_ui32, BitonicSortAVX2) {
    auto begin = V.data();
    gcsort::smallsort::bitonic<uint32_t, vector_machine::AVX2>::sort(begin, GetParam());
    EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

INSTANTIATE_TEST_SUITE_P(Bitonic,
                         BitonicAVX512_ui32,
                         ValuesIn(range(16, 256, 16)),
                         PrintValue());

TEST_P(BitonicAVX512_ui32, BitonicSortAVX512) {
  auto begin = V.data();
  gcsort::smallsort::bitonic<uint32_t, vector_machine::AVX512>::sort(begin, GetParam());
  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

}