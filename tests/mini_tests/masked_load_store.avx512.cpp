#include "vxsort_targets_enable_avx512.h"

#include <vector_machine/machine_traits.avx512.h>

#include "masked_load_store_test.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

template<typename T>
using AVX512MaskedLoadStoreTest = PageWithLavaBoundariesFixture<T, VM::AVX512>;

using MaskedTypes = ::testing::Types<i32, u32, i64, u64>;
TYPED_TEST_SUITE(AVX512MaskedLoadStoreTest, MaskedTypes);

TYPED_TEST(AVX512MaskedLoadStoreTest, PrefixLoadOnPageBoundaryWorks) {
    test_prefix_mask_load_on_page_boundary<TypeParam , VM::AVX512>(this);
}

TYPED_TEST(AVX512MaskedLoadStoreTest, SuffixLoadOnPageBoundaryWorks) {
    test_suffix_mask_load_on_page_boundary<TypeParam , VM::AVX512>(this);
}

TYPED_TEST(AVX512MaskedLoadStoreTest, LeftAlignmentWorks) {
    test_left_alignment_and_masked_loads<TypeParam , VM::AVX512>(this);
}

TYPED_TEST(AVX512MaskedLoadStoreTest, RightAlignmentWorks) {
    test_right_alignment_and_masked_loads<TypeParam , VM::AVX512>(this);
}

};

#include "vxsort_targets_disable.h"
