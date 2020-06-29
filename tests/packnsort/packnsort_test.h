#ifndef VXSORT_FULLSORT_TEST_H
#define VXSORT_FULLSORT_TEST_H

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <vector>
#include <isa_detection.h>
#include <packer.h>
#include <vxsort.h>


namespace vxsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using vxsort::vector_machine;

template <int Unroll, vector_machine M, int Shift>
void perform_packedvxsort_test(std::vector<int64_t> V) {
    if (!vxsort::supports_vector_machine(M)) {
        GTEST_SKIP_("Current CPU does not support the minimal features for this test");
        return;
    }
    auto begin = (int32_t *) V.data();
    auto end = ((int32_t *) V.data()) + V.size() - 1;

    auto sorter = vxsort::vxsort<int32_t, M, Unroll>();
    vxsort::packer<int64_t, int32_t, M, Shift, 2>::pack(V.data(), V.size(), 0x1000);
    sorter.sort(begin, end);
    vxsort::packer<int64_t, int32_t, M, Shift, 2>::unpack((int32_t *) V.data(), V.size(), 0x1000);

    EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

}

#endif  // VXSORT_FULLSORT_TEST_H
