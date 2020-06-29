#include "vxsort_targets_enable_avx2.h"

#include "gtest/gtest.h"

#include <smallsort/avx2/bitonic_machine.avx2.h>
#include <vector_machine/machine_traits.avx2.h>

#include "fullsort_test.h"
#include "../fixtures.h"

namespace vxsort_tests {
using namespace vxsort::types;
using testing::Types;

using namespace vxsort;

struct FullSortTestAVX2_i32 : public SortWithSlackTest<i32> {};
struct FullSortTestAVX2_u32 : public SortWithSlackTest<u32> {};
struct FullSortTestAVX2_i64 : public SortWithSlackTest<i64> {};
struct FullSortTestAVX2_u64 : public SortWithSlackTest<u64> {};
struct FullSortTestAVX2_f32 : public SortWithSlackTest<f32> {};
struct FullSortTestAVX2_f64 : public SortWithSlackTest<f64> {};

auto vxsort_i32_params_avx2 = ValuesIn(SizeAndSlack<i32>::generate(10, 1000000, 10, 16, 0x1000, 0x1));
auto vxsort_u32_params_avx2 = ValuesIn(SizeAndSlack<u32>::generate(10, 1000000, 10, 16, 0x1000, 0x1));
auto vxsort_f32_params_avx2 = ValuesIn(SizeAndSlack<f32>::generate(10, 1000000, 10, 16, 1234.5, 0.1f));
auto vxsort_i64_params_avx2 = ValuesIn(SizeAndSlack<i64>::generate(10, 1000000, 10, 8, 0x1000, 0x1));
auto vxsort_u64_params_avx2 = ValuesIn(SizeAndSlack<u64>::generate(10, 1000000, 10, 8, 0x1000, 0x1));
auto vxsort_f64_params_avx2 = ValuesIn(SizeAndSlack<f64>::generate(10, 1000000, 10, 8, 1234.5, 0.1));

INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_i32, vxsort_i32_params_avx2, PrintSizeAndSlack<i32>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_u32, vxsort_u32_params_avx2, PrintSizeAndSlack<u32>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_f32, vxsort_f32_params_avx2, PrintSizeAndSlack<f32>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_i64, vxsort_i64_params_avx2, PrintSizeAndSlack<i64>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_u64, vxsort_u64_params_avx2, PrintSizeAndSlack<u64>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_f64, vxsort_f64_params_avx2, PrintSizeAndSlack<f64>());

TEST_P(FullSortTestAVX2_i32, VxSortAVX2_1)  { vxsort_test<i32,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i32, VxSortAVX2_2)  { vxsort_test<i32,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i32, VxSortAVX2_4)  { vxsort_test<i32,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i32, VxSortAVX2_8)  { vxsort_test<i32,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i32, VxSortAVX2_12) { vxsort_test<i32, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTestAVX2_u32, VxSortAVX2_1)  { vxsort_test<u32,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_u32, VxSortAVX2_2)  { vxsort_test<u32,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_u32, VxSortAVX2_4)  { vxsort_test<u32,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_u32, VxSortAVX2_8)  { vxsort_test<u32,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_u32, VxSortAVX2_12) { vxsort_test<u32, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTestAVX2_f32, VxSortAVX2_1)  { vxsort_test<f32,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_f32, VxSortAVX2_2)  { vxsort_test<f32,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_f32, VxSortAVX2_4)  { vxsort_test<f32,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_f32, VxSortAVX2_8)  { vxsort_test<f32,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_f32, VxSortAVX2_12) { vxsort_test<f32, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTestAVX2_i64, VxSortAVX2_1)  { vxsort_test<i64,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i64, VxSortAVX2_2)  { vxsort_test<i64,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i64, VxSortAVX2_4)  { vxsort_test<i64,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i64, VxSortAVX2_8)  { vxsort_test<i64,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i64, VxSortAVX2_12) { vxsort_test<i64, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTestAVX2_u64, VxSortAVX2_1)  { vxsort_test<u64,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_u64, VxSortAVX2_2)  { vxsort_test<u64,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_u64, VxSortAVX2_4)  { vxsort_test<u64,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_u64, VxSortAVX2_8)  { vxsort_test<u64,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_u64, VxSortAVX2_12) { vxsort_test<u64, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTestAVX2_f64, VxSortAVX2_1)  { vxsort_test<f64,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_f64, VxSortAVX2_2)  { vxsort_test<f64,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_f64, VxSortAVX2_4)  { vxsort_test<f64,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_f64, VxSortAVX2_8)  { vxsort_test<f64,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_f64, VxSortAVX2_12) { vxsort_test<f64, 12, vector_machine::AVX2>(V); }

struct FullSortStridedTestAVX2_i64 : public SortWithStrideTest<i64> {};
auto vxsort_int64_stride_params_avx2  = ValuesIn(SizeAndStride<i64>::generate(1000000, 0x8L, 0x4000000L, 0x80000000L));
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortStridedTestAVX2_i64, vxsort_int64_stride_params_avx2, PrintSizeAndStride<i64>());

TEST_P(FullSortStridedTestAVX2_i64, VxSortStridedAVX2_1)  { vxsort_hinted_test<i64,  1, 3, vector_machine::AVX2>(V, MinValue, MaxValue); }
TEST_P(FullSortStridedTestAVX2_i64, VxSortStridedAVX2_2)  { vxsort_hinted_test<i64,  2, 3, vector_machine::AVX2>(V, MinValue, MaxValue); }
TEST_P(FullSortStridedTestAVX2_i64, VxSortStridedAVX2_4)  { vxsort_hinted_test<i64,  4, 3, vector_machine::AVX2>(V, MinValue, MaxValue); }
TEST_P(FullSortStridedTestAVX2_i64, VxSortStridedAVX2_8)  { vxsort_hinted_test<i64,  8, 3, vector_machine::AVX2>(V, MinValue, MaxValue); }
TEST_P(FullSortStridedTestAVX2_i64, VxSortStridedAVX2_12) { vxsort_hinted_test<i64, 12, 3, vector_machine::AVX2>(V, MinValue, MaxValue); }

}

#include "vxsort_targets_disable.h"
