#include "vxsort_targets_enable_avx2.h"

#include <vector_machine/machine_traits.avx2.h>

#include "masked_load_store_test.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

template<typename T>
using AVX2MaskedLoadStoreTest = PageWithLavaBoundariesFixture<T, VM::AVX2>;

using MaskedTypes = ::testing::Types<i32, u32, i64, u64>;
TYPED_TEST_SUITE(AVX2MaskedLoadStoreTest, MaskedTypes);

TYPED_TEST(AVX2MaskedLoadStoreTest, PrefixLoadOnPageBoundaryWorks) {
    test_prefix_mask_load_on_page_boundary<TypeParam , VM::AVX2>(this);
}

TYPED_TEST(AVX2MaskedLoadStoreTest, SuffixLoadOnPageBoundaryWorks) {
    test_suffix_mask_load_on_page_boundary<TypeParam , VM::AVX2>(this);
}

TYPED_TEST(AVX2MaskedLoadStoreTest, LeftAlignmentWorks) {
    test_left_alignment_and_masked_loads<TypeParam , VM::AVX2>(this);
}

TYPED_TEST(AVX2MaskedLoadStoreTest, RightAlignmentWorks) {
    test_right_alignment_and_masked_loads<TypeParam , VM::AVX2>(this);
}


};

#include "vxsort_targets_disable.h"