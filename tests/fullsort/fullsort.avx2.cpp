#include "vxsort_targets_enable_avx2.h"

#include "gtest/gtest.h"
#include "../fixtures.h"

#include <smallsort/avx2/bitonic_machine.AVX2.f64.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.f32.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.i32.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.i64.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.u32.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.u64.generated.h>
#include <vector_machine/machine_traits.avx2.h>

#include "fullsort_test.h"

namespace vxsort_tests {
using testing::Types;

using namespace vxsort;

struct FullSortTestAVX2_i32 : public SortWithSlackTest<i32> {};
struct FullSortTestAVX2_ui32 : public SortWithSlackTest<u32> {};
struct FullSortTestAVX2_i64 : public SortWithSlackTest<i64> {};
struct FullSortTestAVX2_ui64 : public SortWithSlackTest<u64> {};
struct FullSortTestAVX2_float : public SortWithSlackTest<float> {};
struct FullSortTestAVX2_double : public SortWithSlackTest<double> {};

auto vxsort_int32_params_avx2  = ValuesIn(SizeAndSlack<i32>::generate(10, 1000000, 10, 16, 0x1000, 0x1));
auto vxsort_uint32_params_avx2 = ValuesIn(SizeAndSlack<u32>::generate(10, 1000000, 10, 16, 0x1000, 0x1));
auto vxsort_float_params_avx2  = ValuesIn(SizeAndSlack<float>::generate(10, 1000000, 10, 16, 1234.5, 0.1f));
auto vxsort_int64_params_avx2  = ValuesIn(SizeAndSlack<i64>::generate(10, 1000000, 10, 8, 0x1000, 0x1));
auto vxsort_uint64_params_avx2 = ValuesIn(SizeAndSlack<u64>::generate(10, 1000000, 10, 8, 0x1000, 0x1));
auto vxsort_double_params_avx2 = ValuesIn(SizeAndSlack<double>::generate(10, 1000000, 10, 8, 1234.5, 0.1));

INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_i32,    vxsort_int32_params_avx2, PrintSizeAndSlack<i32>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_ui32,   vxsort_uint32_params_avx2, PrintSizeAndSlack<u32>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_float,  vxsort_float_params_avx2, PrintSizeAndSlack<float>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_i64,    vxsort_int64_params_avx2, PrintSizeAndSlack<i64>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_ui64,   vxsort_uint64_params_avx2, PrintSizeAndSlack<u64>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_double, vxsort_double_params_avx2, PrintSizeAndSlack<double>());

TEST_P(FullSortTestAVX2_i32, VxSortAVX2_1)  { perform_vxsort_test<i32,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i32, VxSortAVX2_2)  { perform_vxsort_test<i32,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i32, VxSortAVX2_4)  { perform_vxsort_test<i32,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i32, VxSortAVX2_8)  { perform_vxsort_test<i32,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i32, VxSortAVX2_12) { perform_vxsort_test<i32, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTestAVX2_ui32, VxSortAVX2_1)  { perform_vxsort_test<u32,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_ui32, VxSortAVX2_2)  { perform_vxsort_test<u32,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_ui32, VxSortAVX2_4)  { perform_vxsort_test<u32,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_ui32, VxSortAVX2_8)  { perform_vxsort_test<u32,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_ui32, VxSortAVX2_12) { perform_vxsort_test<u32, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTestAVX2_float, VxSortAVX2_1)  { perform_vxsort_test<float,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_float, VxSortAVX2_2)  { perform_vxsort_test<float,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_float, VxSortAVX2_4)  { perform_vxsort_test<float,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_float, VxSortAVX2_8)  { perform_vxsort_test<float,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_float, VxSortAVX2_12) { perform_vxsort_test<float, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTestAVX2_i64, VxSortAVX2_1)  { perform_vxsort_test<i64,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i64, VxSortAVX2_2)  { perform_vxsort_test<i64,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i64, VxSortAVX2_4)  { perform_vxsort_test<i64,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i64, VxSortAVX2_8)  { perform_vxsort_test<i64,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i64, VxSortAVX2_12) { perform_vxsort_test<i64, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTestAVX2_ui64, VxSortAVX2_1)  { perform_vxsort_test<u64,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_ui64, VxSortAVX2_2)  { perform_vxsort_test<u64,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_ui64, VxSortAVX2_4)  { perform_vxsort_test<u64,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_ui64, VxSortAVX2_8)  { perform_vxsort_test<u64,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_ui64, VxSortAVX2_12) { perform_vxsort_test<u64, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTestAVX2_double, VxSortAVX2_1)  { perform_vxsort_test<double,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_double, VxSortAVX2_2)  { perform_vxsort_test<double,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_double, VxSortAVX2_4)  { perform_vxsort_test<double,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_double, VxSortAVX2_8)  { perform_vxsort_test<double,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_double, VxSortAVX2_12) { perform_vxsort_test<double, 12, vector_machine::AVX2>(V); }

struct FullSortStridedTestAVX2_i64 : public SortWithStrideTest<i64> {};
auto vxsort_int64_stride_params_avx2  = ValuesIn(SizeAndStride<i64>::generate(1000000, 0x8L, 0x4000000L, 0x80000000L));
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortStridedTestAVX2_i64, vxsort_int64_stride_params_avx2, PrintSizeAndStride<i64>());

TEST_P(FullSortStridedTestAVX2_i64, VxSortStridedAVX2_1)  { perform_vxsort_hinted_test<i64,  1, 3, vector_machine::AVX2>(V, MinValue, MaxValue); }
TEST_P(FullSortStridedTestAVX2_i64, VxSortStridedAVX2_2)  { perform_vxsort_hinted_test<i64,  2, 3, vector_machine::AVX2>(V, MinValue, MaxValue); }
TEST_P(FullSortStridedTestAVX2_i64, VxSortStridedAVX2_4)  { perform_vxsort_hinted_test<i64,  4, 3, vector_machine::AVX2>(V, MinValue, MaxValue); }
TEST_P(FullSortStridedTestAVX2_i64, VxSortStridedAVX2_8)  { perform_vxsort_hinted_test<i64,  8, 3, vector_machine::AVX2>(V, MinValue, MaxValue); }
TEST_P(FullSortStridedTestAVX2_i64, VxSortStridedAVX2_12) { perform_vxsort_hinted_test<i64, 12, 3, vector_machine::AVX2>(V, MinValue, MaxValue); }

}

#include "vxsort_targets_disable.h"
