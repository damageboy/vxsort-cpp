#include "vxsort_targets_enable_avx2.h"


#include "gtest/gtest.h"

#include <vector_machine/machine_traits.avx2.h>

#include "packunpack_test.h"
#include "../fixtures.h"

namespace vxsort_tests {

using vxsort::vector_machine;
using namespace vxsort::types;

struct PositiveTestAVX2_i32 : public SortWithSlackTest<i32> {};
struct PositiveTestAVX2_i64 : public SortWithSlackTest<i64> {};

struct NegativeTestAVX2_i32 : public SortWithSlackTest<i32> {};
struct NegativeTestAVX2_i64 : public SortWithSlackTest<i64> {};

const u32 positive_base_32 = 2LL<<17;
const u32 negative_base_32 = -(2LL<<17);
auto packunpack_positive_values_avx2_32 = ValuesIn(SizeAndSlack<i32>::generate(10, 10000, 10, 8, positive_base_32, 0x1, false));
auto packunpack_negative_values_avx2_32 = ValuesIn(SizeAndSlack<i32>::generate(10, 10000, 10, 8, negative_base_32, 0x1, false));

INSTANTIATE_TEST_SUITE_P(PackUnpack, PositiveTestAVX2_i32, packunpack_positive_values_avx2_32, PrintSizeAndSlack<i32>());
INSTANTIATE_TEST_SUITE_P(PackUnpack, NegativeTestAVX2_i32, packunpack_negative_values_avx2_32, PrintSizeAndSlack<i32>());

TEST_P(PositiveTestAVX2_i32, PackUnpackPositive1_U1) { packunpack_test<i32, vector_machine::AVX2, 1>(V, positive_base_32); }
TEST_P(PositiveTestAVX2_i32, PackUnpackPositive1_U2) { packunpack_test<i32, vector_machine::AVX2, 2>(V, positive_base_32); }
TEST_P(PositiveTestAVX2_i32, PackUnpackPositive1_U4) { packunpack_test<i32, vector_machine::AVX2, 4>(V, positive_base_32); }

TEST_P(NegativeTestAVX2_i32, PackUnpackPositive1_U1) { packunpack_test<i32, vector_machine::AVX2, 1>(V, negative_base_32); }
TEST_P(NegativeTestAVX2_i32, PackUnpackPositive1_U2) { packunpack_test<i32, vector_machine::AVX2, 2>(V, negative_base_32); }
TEST_P(NegativeTestAVX2_i32, PackUnpackPositive1_U4) { packunpack_test<i32, vector_machine::AVX2, 4>(V, negative_base_32); }

const u64 positive_base_64 = 2LL<<33;
const u64 negative_base_64 = -(2LL<<33);
auto packunpack_positive_values_avx2_64 = ValuesIn(SizeAndSlack<i64>::generate(10, 1000000, 10, 8, positive_base_64, 0x8, false));
auto packunpack_negative_values_avx2_64 = ValuesIn(SizeAndSlack<i64>::generate(10, 1000000, 10, 8, negative_base_64, 0x8, false));

INSTANTIATE_TEST_SUITE_P(PackUnpack, PositiveTestAVX2_i64, packunpack_positive_values_avx2_64, PrintSizeAndSlack<i64>());
INSTANTIATE_TEST_SUITE_P(PackUnpack, NegativeTestAVX2_i64, packunpack_negative_values_avx2_64, PrintSizeAndSlack<i64>());

TEST_P(PositiveTestAVX2_i64, PackUnpackPositive1_U1) { packunpack_test<i64, vector_machine::AVX2, 1, 3>(V, positive_base_64); }
TEST_P(PositiveTestAVX2_i64, PackUnpackPositive1_U2) { packunpack_test<i64, vector_machine::AVX2, 2, 3>(V, positive_base_64); }
TEST_P(PositiveTestAVX2_i64, PackUnpackPositive1_U4) { packunpack_test<i64, vector_machine::AVX2, 4, 3>(V, positive_base_64); }

TEST_P(NegativeTestAVX2_i64, PackUnpackPositive1_U1) { packunpack_test<i64, vector_machine::AVX2, 1, 3>(V, negative_base_64); }
TEST_P(NegativeTestAVX2_i64, PackUnpackPositive1_U2) { packunpack_test<i64, vector_machine::AVX2, 2, 3>(V, negative_base_64); }
TEST_P(NegativeTestAVX2_i64, PackUnpackPositive1_U4) { packunpack_test<i64, vector_machine::AVX2, 4, 3>(V, negative_base_64); }

}
#include "vxsort_targets_disable.h"
