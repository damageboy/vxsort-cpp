#ifndef VXSORT_PACKUNPACK_TEST_H
#define VXSORT_PACKUNPACK_TEST_H


#include "../fixtures.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <isa_detection.h>
#include <packer.h>

namespace vxsort_tests {

using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using vxsort::vector_machine;

template <vector_machine M, int Shift, int Unroll, int MinLength, bool RespectPackingOrder>
void perform_packunpack_test(std::vector<int64_t> V, int64_t base) {

  if (!vxsort::supports_vector_machine(M)) {
    GTEST_SKIP_(
        "Current CPU does not support the minimal features for this test");
    return;
  }

  auto copy = std::vector<int64_t>(V.size());
  copy.assign(V.begin(), V.end());

  auto begin = copy.data();
  vxsort::packer<int64_t, int32_t, M, Shift, Unroll, MinLength, RespectPackingOrder>::pack(begin, copy.size(), base);
  vxsort::packer<int64_t, int32_t, M, Shift, Unroll, MinLength>::unpack((int32_t *) begin, copy.size(), base);

  // Sort with std::sort for cases where packing change order
  if (!RespectPackingOrder) {
    std::sort(copy.begin(), copy.end());
  }


  EXPECT_THAT(copy, ElementsAreArray(V));
}

}

#endif  // VXSORT_PACKUNPACK_TEST_H
