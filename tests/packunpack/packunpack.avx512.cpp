#include "vxsort_targets_enable_avx512.h"

#include "gtest/gtest.h"

#include <vector_machine/machine_traits.avx512.h>

#include "packunpack_test.h"
#include "../fixtures.h"

namespace vxsort_tests {

using vxsort::vector_machine;
using namespace vxsort::types;

struct PositiveTestAVX512_i32 : public SortWithSlackTest<i32> {};
struct PositiveTestAVX512_i64 : public SortWithSlackTest<i64> {};

struct NegativeTestAVX512_i32 : public SortWithSlackTest<i32> {};
struct NegativeTestAVX512_i64 : public SortWithSlackTest<i64> {};

const u32 positive_base_32 = 2LL<<17;
const u32 negative_base_32 = -(2LL<<17);
auto packunpack_positive_values_avx512_32 = ValuesIn(SizeAndSlack<i32>::generate(10, 10000, 10, 16, positive_base_32, 0x1, false));
auto packunpack_negative_values_avx512_32 = ValuesIn(SizeAndSlack<i32>::generate(10, 10000, 10, 16, negative_base_32, 0x1, false));
INSTANTIATE_TEST_SUITE_P(PackUnpack, PositiveTestAVX512_i32, packunpack_positive_values_avx512_32, PrintSizeAndSlack<i32>());
INSTANTIATE_TEST_SUITE_P(PackUnpack, NegativeTestAVX512_i32, packunpack_negative_values_avx512_32, PrintSizeAndSlack<i32>());

TEST_P(PositiveTestAVX512_i32, PackUnpackPositive_U1) { packunpack_test<i32, vector_machine::AVX512, 1>(V, positive_base_32); }
TEST_P(PositiveTestAVX512_i32, PackUnpackPositive_U2) { packunpack_test<i32, vector_machine::AVX512, 2>(V, positive_base_32); }
TEST_P(PositiveTestAVX512_i32, PackUnpackPositive_U4) { packunpack_test<i32, vector_machine::AVX512, 4>(V, positive_base_32); }

TEST_P(NegativeTestAVX512_i32, PackUnpackNegative_U1) { packunpack_test<i32, vector_machine::AVX512, 1>(V, negative_base_32); }
TEST_P(NegativeTestAVX512_i32, PackUnpackPositive_U2) { packunpack_test<i32, vector_machine::AVX512, 2>(V, negative_base_32); }
TEST_P(NegativeTestAVX512_i32, PackUnpackPositive_U4) { packunpack_test<i32, vector_machine::AVX512, 4>(V, negative_base_32); }

const u64 positive_base_64 = 2LL<<33;
const u64 negative_base_64 = -(2LL<<33);
auto packunpack_positive_values_avx512_64 = ValuesIn(SizeAndSlack<int64_t>::generate(10, 1000000, 10, 16, positive_base_64, 0x8, false));
auto packunpack_negative_values_avx512_64 = ValuesIn(SizeAndSlack<int64_t>::generate(10, 1000000, 10, 16, negative_base_64, 0x8, false));
INSTANTIATE_TEST_SUITE_P(PackUnpackAVX512, PositiveTestAVX512_i64, packunpack_positive_values_avx512_64, PrintSizeAndSlack<i64>());
INSTANTIATE_TEST_SUITE_P(PackUnpackAVX512, NegativeTestAVX512_i64, packunpack_negative_values_avx512_64, PrintSizeAndSlack<i64>());

TEST_P(PositiveTestAVX512_i64, PackUnpackPositive_U1) { packunpack_test<i64, vector_machine::AVX512, 1, 3>(V, positive_base_64); }
TEST_P(PositiveTestAVX512_i64, PackUnpackPositive_U2) { packunpack_test<i64, vector_machine::AVX512, 2, 3>(V, positive_base_64); }
TEST_P(PositiveTestAVX512_i64, PackUnpackPositive_U4) { packunpack_test<i64, vector_machine::AVX512, 3, 3>(V, positive_base_64); }

TEST_P(NegativeTestAVX512_i64, PackUnpackNegative_U1) { packunpack_test<i64, vector_machine::AVX512, 1, 3>(V, negative_base_64); }
TEST_P(NegativeTestAVX512_i64, PackUnpackPositive_U2) { packunpack_test<i64, vector_machine::AVX512, 2, 3>(V, negative_base_64); }
TEST_P(NegativeTestAVX512_i64, PackUnpackPositive_U4) { packunpack_test<i64, vector_machine::AVX512, 4, 3>(V, negative_base_64); }

}
#include "vxsort_targets_disable.h"
