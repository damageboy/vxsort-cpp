#include "vxsort_targets_enable_avx2.h"

#include "masked_load_store.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

template <typename T>
using MaskedLoadStoreDeathTest = MaskedLoadStoreTest<T, VM::NONE>;

using MaskedTypes = ::testing::Types<i32>;
TYPED_TEST_SUITE(MaskedLoadStoreDeathTest, MaskedTypes);

TYPED_TEST(MaskedLoadStoreDeathTest, IsSane) {
    EXPECT_EQ(*this->page_with_data, this->get_expected_value(this->page_with_data));
}


TYPED_TEST(MaskedLoadStoreDeathTest, WhatKillsMeMakesMeStronger1) {
    ASSERT_DEATH(*((volatile i32 *) this->page_with_data - 1), "");
}

TYPED_TEST(MaskedLoadStoreDeathTest, WhatKillsMeMakesMeStronger2) {
    ASSERT_DEATH(*((volatile i32 *) this->page_with_data + this->num_elements), "");
}

};

#include "vxsort_targets_disable.h"
