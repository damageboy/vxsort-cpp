#include "vxsort_targets_enable_avx512.h"

#include "gtest/gtest.h"
#include "../fixtures.h"

#include <smallsort/avx512/bitonic_machine.AVX512.f64.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.f32.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.i32.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.i64.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.u32.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.u64.generated.h>
#include <vector_machine/machine_traits.avx512.h>
#include "fullsort_test.h"

namespace vxsort_tests {
using testing::Types;

using namespace vxsort;

struct FullSortTestAVX512_i32 : public SortWithSlackTest<i32> {};
struct FullSortTestAVX512_ui32 : public SortWithSlackTest<u32> {};
struct FullSortTestAVX512_i64 : public SortWithSlackTest<i64> {};
struct FullSortTestAVX512_ui64 : public SortWithSlackTest<u64> {};
struct FullSortTestAVX512_float : public SortWithSlackTest<float> {};
struct FullSortTestAVX512_double : public SortWithSlackTest<double> {};

auto vxsort_int32_params_avx512 = ValuesIn(SizeAndSlack<i32>::generate(10, 1000000, 10, 32, 0x1000, 0x1));
auto vxsort_uint32_params_avx512 = ValuesIn(SizeAndSlack<u32>::generate(10, 1000000, 10, 32, 0x1000, 0x1));
auto vxsort_float_params_avx512 = ValuesIn(SizeAndSlack<float>::generate(10, 1000000, 10, 32, 1234.5, 0.1f));
auto vxsort_int64_params_avx512 = ValuesIn(SizeAndSlack<i64>::generate(10, 1000000, 10, 16, 0x1000, 0x1));
auto vxsort_uint64_params_avx512 = ValuesIn(SizeAndSlack<u64>::generate(10, 1000000, 10, 16, 0x1000, 0x1));
auto vxsort_double_params_avx512 = ValuesIn(SizeAndSlack<double>::generate(10, 1000000, 10, 16, 0x1000, 0x1));

INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_i32,    vxsort_int32_params_avx512, PrintSizeAndSlack<i32>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_ui32,   vxsort_uint32_params_avx512, PrintSizeAndSlack<u32>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_float,  vxsort_float_params_avx512, PrintSizeAndSlack<float>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_i64,    vxsort_int64_params_avx512, PrintSizeAndSlack<i64>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_ui64,   vxsort_uint64_params_avx512, PrintSizeAndSlack<u64>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_double, vxsort_double_params_avx512, PrintSizeAndSlack<double>());

TEST_P(FullSortTestAVX512_i32, VxSortAVX512_1)  { perform_vxsort_test<i32,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i32, VxSortAVX512_2)  { perform_vxsort_test<i32,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i32, VxSortAVX512_4)  { perform_vxsort_test<i32,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i32, VxSortAVX512_8)  { perform_vxsort_test<i32,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i32, VxSortAVX512_12) { perform_vxsort_test<i32, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_ui32, VxSortAVX512_1)  { perform_vxsort_test<u32,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_ui32, VxSortAVX512_2)  { perform_vxsort_test<u32,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_ui32, VxSortAVX512_4)  { perform_vxsort_test<u32,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_ui32, VxSortAVX512_8)  { perform_vxsort_test<u32,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_ui32, VxSortAVX512_12) { perform_vxsort_test<u32, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_float, VxSortAVX512_1)  { perform_vxsort_test<float,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_float, VxSortAVX512_2)  { perform_vxsort_test<float,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_float, VxSortAVX512_4)  { perform_vxsort_test<float,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_float, VxSortAVX512_8)  { perform_vxsort_test<float,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_float, VxSortAVX512_12) { perform_vxsort_test<float, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_i64, VxSortAVX512_1)  { perform_vxsort_test<i64,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i64, VxSortAVX512_2)  { perform_vxsort_test<i64,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i64, VxSortAVX512_4)  { perform_vxsort_test<i64,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i64, VxSortAVX512_8)  { perform_vxsort_test<i64,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i64, VxSortAVX512_12) { perform_vxsort_test<i64, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_ui64, VxSortAVX512_1)  { perform_vxsort_test<u64,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_ui64, VxSortAVX512_2)  { perform_vxsort_test<u64,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_ui64, VxSortAVX512_4)  { perform_vxsort_test<u64,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_ui64, VxSortAVX512_8)  { perform_vxsort_test<u64,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_ui64, VxSortAVX512_12) { perform_vxsort_test<u64, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_double, VxSortAVX512_1)  { perform_vxsort_test<double,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_double, VxSortAVX512_2)  { perform_vxsort_test<double,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_double, VxSortAVX512_4)  { perform_vxsort_test<double,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_double, VxSortAVX512_8)  { perform_vxsort_test<double,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_double, VxSortAVX512_12) { perform_vxsort_test<double, 12, vector_machine::AVX512>(V); }

struct FullSortStridedTestAVX512_i64 : public SortWithStrideTest<i64> {};
auto vxsort_int64_stride_params_avx512  = ValuesIn(SizeAndStride<i64>::generate(1000000, 0x8L, 0x1000000L, 0x80000000L));
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortStridedTestAVX512_i64, vxsort_int64_stride_params_avx512, PrintSizeAndStride<i64>());

TEST_P(FullSortStridedTestAVX512_i64, VxSortStridedAVX512_1)  { perform_vxsort_hinted_test<i64,  1, 3, vector_machine::AVX512>(V, MinValue, MaxValue); }
TEST_P(FullSortStridedTestAVX512_i64, VxSortStridedAVX512_2)  { perform_vxsort_hinted_test<i64,  2, 3, vector_machine::AVX512>(V, MinValue, MaxValue); }
TEST_P(FullSortStridedTestAVX512_i64, VxSortStridedAVX512_4)  { perform_vxsort_hinted_test<i64,  4, 3, vector_machine::AVX512>(V, MinValue, MaxValue); }
TEST_P(FullSortStridedTestAVX512_i64, VxSortStridedAVX512_8)  { perform_vxsort_hinted_test<i64,  8, 3, vector_machine::AVX512>(V, MinValue, MaxValue); }
TEST_P(FullSortStridedTestAVX512_i64, VxSortStridedAVX512_12) { perform_vxsort_hinted_test<i64, 12, 3, vector_machine::AVX512>(V, MinValue, MaxValue); }


}

#include "vxsort_targets_disable.h"
