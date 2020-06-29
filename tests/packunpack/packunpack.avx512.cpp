#include "vxsort_targets_enable_avx512.h"

#include "gtest/gtest.h"
#include "../fixtures.h"

#include <machine_traits.avx512.h>

#include "packunpack_test.h"

namespace vxsort_tests {

using vxsort::vector_machine;

struct PackUnpackPositiveTestAVX512_i64 : public SortWithSlackTest<int64_t> {};

struct PackUnpackFullTestAVX512_i64 : public SortWithSlackTest<int64_t> {};

const uint64_t positive_base = 2LL<<33;
const uint64_t negative_base = -(2LL<<33);

auto packunpack_positive_values_avx512_64 = ValuesIn(SizeAndSlack<int64_t>::generate(10, 1000000, 10, 16, positive_base, 0x8, false));
auto packunpack_values_avx512_64 = ValuesIn(SizeAndSlack<int64_t>::generate(10, 1000000, 10, 16,  negative_base, 0x8, false));


INSTANTIATE_TEST_SUITE_P(PackUnpackAVX512, PackUnpackPositiveTestAVX512_i64, packunpack_positive_values_avx512_64, PrintSizeAndSlack<int64_t>());
INSTANTIATE_TEST_SUITE_P(PackUnpackAVX512, PackUnpackFullTestAVX512_i64,     packunpack_values_avx512_64, PrintSizeAndSlack<int64_t>());

TEST_P(PackUnpackPositiveTestAVX512_i64, PackUnpackPositive1_U1) { perform_packunpack_test<vector_machine::AVX512, 3, 1, 1, true>(V, positive_base); }
TEST_P(PackUnpackPositiveTestAVX512_i64, PackUnpackPositive2_U1) { perform_packunpack_test<vector_machine::AVX512, 3, 1, 1, false>(V, positive_base); }

TEST_P(PackUnpackPositiveTestAVX512_i64, PackUnpackPositive1_U2) { perform_packunpack_test<vector_machine::AVX512, 3, 2, 1, true>(V, positive_base); }
TEST_P(PackUnpackPositiveTestAVX512_i64, PackUnpackPositive2_U2) { perform_packunpack_test<vector_machine::AVX512, 3, 2, 1, false>(V, positive_base); }

TEST_P(PackUnpackPositiveTestAVX512_i64, PackUnpackPositive1_U4) { perform_packunpack_test<vector_machine::AVX512, 3, 4, 1, true>(V, positive_base); }
TEST_P(PackUnpackPositiveTestAVX512_i64, PackUnpackPositive2_U4) { perform_packunpack_test<vector_machine::AVX512, 3, 4, 1, false>(V, positive_base); }

TEST_P(PackUnpackFullTestAVX512_i64, PackUnpackPositive1) { perform_packunpack_test<vector_machine::AVX512, 3, 1, 1, true>(V, negative_base); }
TEST_P(PackUnpackFullTestAVX512_i64, PackUnpackPositive2) { perform_packunpack_test<vector_machine::AVX512, 3, 1, 1, false>(V, negative_base); }


}
#include "vxsort_targets_disable.h"
