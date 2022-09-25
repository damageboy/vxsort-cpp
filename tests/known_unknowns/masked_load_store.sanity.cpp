#include "vxsort_targets_enable_avx2.h"

#include "masked_load_store.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

template<typename T>
using MaskedLoadStoreDeathTest = MaskedLoadStoreTest<T>;

using MaskedTypes = ::testing::Types<i32>;
TYPED_TEST_SUITE(MaskedLoadStoreDeathTest, MaskedTypes);

TYPED_TEST(MaskedLoadStoreDeathTest, IsSane) {
    EXPECT_EQ(*this->page_with_0x66, 0x66666666);
}

TYPED_TEST(MaskedLoadStoreDeathTest, WhatKillsMeMakesMeStronger1) {
    EXPECT_EXIT(*((volatile i32 *) this->page_with_0x66 - 1), testing::KilledBySignal(11), "");
}

TYPED_TEST(MaskedLoadStoreDeathTest, WhatKillsMeMakesMeStronger2) {
    EXPECT_EXIT(*((volatile i32 *) this->page_with_0x66 + (page_size / sizeof(i32))), testing::KilledBySignal(11), "");
}

};

#include "vxsort_targets_disable.h"
