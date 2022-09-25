#include "vxsort_targets_enable_avx512.h"

#include "masked_load_store.h"
namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

template<typename T>
using AVX512MaskedLoadStoreTest = MaskedLoadStoreTest<T>;

using MaskedTypes = ::testing::Types<i32, u32, i64, u64>;
TYPED_TEST_SUITE(AVX512MaskedLoadStoreTest, MaskedTypes);

TYPED_TEST(AVX512MaskedLoadStoreTest, PrefixLoadOnPageBoundaryWorks) {
    test_prefix_mask_load_on_page_boundary<TypeParam, VM::AVX512>(this->page_with_0x66, page_size, this->get_expected_value());
}

TYPED_TEST(AVX512MaskedLoadStoreTest, SuffixLoadOnPageBoundaryWorks) {
    test_suffix_mask_load_on_page_boundary<TypeParam , VM::AVX512>(this->page_with_0x66, page_size, this->get_expected_value());
}
};

#include "vxsort_targets_disable.h"
