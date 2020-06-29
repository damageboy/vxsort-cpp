#ifndef VXSORT_PACKUNPACK_TEST_H
#define VXSORT_PACKUNPACK_TEST_H


#include "../fixtures.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <fmt/format.h>

#include <isa_detection.h>
#include <packer.h>

using namespace vxsort::types;

namespace vxsort_tests {

using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using vxsort::vector_machine;
using vxsort::vxsort_machine_traits;

template <typename T, vector_machine M, int Unroll, int Shift=0, int MinLength=1>
void packunpack_test(std::vector<T> V, T base) {
    using VMT = vxsort_machine_traits<T, M>;
    using TPACK = typename VMT::TPACK;
    using Packer = vxsort::packer<T, TPACK, M, Shift, Unroll, MinLength>;

    if (!vxsort::supports_vector_machine(M)) {
        GTEST_SKIP_("Current CPU does not support the minimal features for this test");
        return;
    }

    auto copy = std::vector<T>(V);
    u64 size = copy.size();
    auto begin_packed = Packer::pack(copy.data(), size, base);
    Packer::unpack(begin_packed, size, base);

    std::sort(copy.begin(), copy.end());

    for (auto i = 0ULL; i < size; ++i) {
        if (copy[i] != V[i]) {
            GTEST_FAIL() << fmt::format("Unpacked value at idx #{}  {} != {}", i, copy[i], V[i]);
        }
    }
}

}

#endif  // VXSORT_PACKUNPACK_TEST_H
