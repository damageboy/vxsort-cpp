#include "vxsort_targets_enable_avx2.h"

#include "gtest/gtest.h"

#include <smallsort/avx2/bitonic_machine.avx2.h>
#include <vector_machine/machine_traits.avx2.h>

#include "fullsort_test.h"
#include "../sort_fixtures.h"

namespace vxsort_tests {
using namespace vxsort::types;
using testing::Types;

using VM = vxsort::vector_machine;
using namespace vxsort;

struct VxSortAVX2_i16 : public SortWithSlackFixture<i16> {};
struct VxSortAVX2_i32 : public SortWithSlackFixture<i32> {};
struct VxSortAVX2_u16 : public SortWithSlackFixture<u16> {};
struct VxSortAVX2_u32 : public SortWithSlackFixture<u32> {};
struct VxSortAVX2_i64 : public SortWithSlackFixture<i64> {};
struct VxSortAVX2_u64 : public SortWithSlackFixture<u64> {};
struct VxSortAVX2_f32 : public SortWithSlackFixture<f32> {};
struct VxSortAVX2_f64 : public SortWithSlackFixture<f64> {};

auto vxsort_i16_params_avx2 = ValuesIn(SizeAndSlack<i16>::generate(10, 10000,   10, 32, 0x1000, 0x1));
auto vxsort_i32_params_avx2 = ValuesIn(SizeAndSlack<i32>::generate(10, 1000000, 10, 16, 0x1000, 0x1));
auto vxsort_u16_params_avx2 = ValuesIn(SizeAndSlack<u16>::generate(10, 10000,   10, 32, 0x1000, 0x1));
auto vxsort_u32_params_avx2 = ValuesIn(SizeAndSlack<u32>::generate(10, 1000000, 10, 16, 0x1000, 0x1));
auto vxsort_f32_params_avx2 = ValuesIn(SizeAndSlack<f32>::generate(10, 1000000, 10, 16, 1234.5, 0.1f));
auto vxsort_i64_params_avx2 = ValuesIn(SizeAndSlack<i64>::generate(10, 1000000, 10, 8, 0x1000, 0x1));
auto vxsort_u64_params_avx2 = ValuesIn(SizeAndSlack<u64>::generate(10, 1000000, 10, 8, 0x1000, 0x1));
auto vxsort_f64_params_avx2 = ValuesIn(SizeAndSlack<f64>::generate(10, 1000000, 10, 8, 1234.5, 0.1));

INSTANTIATE_TEST_SUITE_P(VxSort, VxSortAVX2_i16, vxsort_i16_params_avx2, PrintSizeAndSlack<i16>());
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortAVX2_i32, vxsort_i32_params_avx2, PrintSizeAndSlack<i32>());
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortAVX2_u16, vxsort_u16_params_avx2, PrintSizeAndSlack<u16>());
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortAVX2_u32, vxsort_u32_params_avx2, PrintSizeAndSlack<u32>());
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortAVX2_f32, vxsort_f32_params_avx2, PrintSizeAndSlack<f32>());
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortAVX2_i64, vxsort_i64_params_avx2, PrintSizeAndSlack<i64>());
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortAVX2_u64, vxsort_u64_params_avx2, PrintSizeAndSlack<u64>());
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortAVX2_f64, vxsort_f64_params_avx2, PrintSizeAndSlack<f64>());

//TEST_P(VxSortAVX2_i16, VxSortAVX2_1)  { vxsort_test<i16,  1, VM::AVX2>(V); }
//TEST_P(VxSortAVX2_i16, VxSortAVX2_2)  { vxsort_test<i16,  2, VM::AVX2>(V); }
//TEST_P(VxSortAVX2_i16, VxSortAVX2_4)  { vxsort_test<i16,  4, VM::AVX2>(V); }
//TEST_P(VxSortAVX2_i16, VxSortAVX2_8)  { vxsort_test<i16,  8, VM::AVX2>(V); }
//TEST_P(VxSortAVX2_i16, VxSortAVX2_12) { vxsort_test<i16, 12, VM::AVX2>(V); }

TEST_P(VxSortAVX2_i32, VxSortAVX2_1)  { vxsort_test<i32,  1, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i32, VxSortAVX2_2)  { vxsort_test<i32,  2, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i32, VxSortAVX2_4)  { vxsort_test<i32,  4, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i32, VxSortAVX2_8)  { vxsort_test<i32,  8, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i32, VxSortAVX2_12) { vxsort_test<i32, 12, VM::AVX2>(V); }

//TEST_P(VxSortAVX2_u16, VxSortAVX2_1)  { vxsort_test<u16,  1, VM::AVX2>(V); }
//TEST_P(VxSortAVX2_u16, VxSortAVX2_2)  { vxsort_test<u16,  2, VM::AVX2>(V); }
//TEST_P(VxSortAVX2_u16, VxSortAVX2_4)  { vxsort_test<u16,  4, VM::AVX2>(V); }
//TEST_P(VxSortAVX2_u16, VxSortAVX2_8)  { vxsort_test<u16,  8, VM::AVX2>(V); }
//TEST_P(VxSortAVX2_u16, VxSortAVX2_12) { vxsort_test<u16, 12, VM::AVX2>(V); }

TEST_P(VxSortAVX2_u32, VxSortAVX2_1)  { vxsort_test<u32,  1, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u32, VxSortAVX2_2)  { vxsort_test<u32,  2, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u32, VxSortAVX2_4)  { vxsort_test<u32,  4, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u32, VxSortAVX2_8)  { vxsort_test<u32,  8, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u32, VxSortAVX2_12) { vxsort_test<u32, 12, VM::AVX2>(V); }

TEST_P(VxSortAVX2_f32, VxSortAVX2_1)  { vxsort_test<f32,  1, VM::AVX2>(V); }
TEST_P(VxSortAVX2_f32, VxSortAVX2_2)  { vxsort_test<f32,  2, VM::AVX2>(V); }
TEST_P(VxSortAVX2_f32, VxSortAVX2_4)  { vxsort_test<f32,  4, VM::AVX2>(V); }
TEST_P(VxSortAVX2_f32, VxSortAVX2_8)  { vxsort_test<f32,  8, VM::AVX2>(V); }
TEST_P(VxSortAVX2_f32, VxSortAVX2_12) { vxsort_test<f32, 12, VM::AVX2>(V); }

TEST_P(VxSortAVX2_i64, VxSortAVX2_1)  { vxsort_test<i64,  1, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i64, VxSortAVX2_2)  { vxsort_test<i64,  2, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i64, VxSortAVX2_4)  { vxsort_test<i64,  4, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i64, VxSortAVX2_8)  { vxsort_test<i64,  8, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i64, VxSortAVX2_12) { vxsort_test<i64, 12, VM::AVX2>(V); }

TEST_P(VxSortAVX2_u64, VxSortAVX2_1)  { vxsort_test<u64,  1, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u64, VxSortAVX2_2)  { vxsort_test<u64,  2, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u64, VxSortAVX2_4)  { vxsort_test<u64,  4, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u64, VxSortAVX2_8)  { vxsort_test<u64,  8, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u64, VxSortAVX2_12) { vxsort_test<u64, 12, VM::AVX2>(V); }

TEST_P(VxSortAVX2_f64, VxSortAVX2_1)  { vxsort_test<f64,  1, VM::AVX2>(V); }
TEST_P(VxSortAVX2_f64, VxSortAVX2_2)  { vxsort_test<f64,  2, VM::AVX2>(V); }
TEST_P(VxSortAVX2_f64, VxSortAVX2_4)  { vxsort_test<f64,  4, VM::AVX2>(V); }
TEST_P(VxSortAVX2_f64, VxSortAVX2_8)  { vxsort_test<f64,  8, VM::AVX2>(V); }
TEST_P(VxSortAVX2_f64, VxSortAVX2_12) { vxsort_test<f64, 12, VM::AVX2>(V); }

struct VxSortWithStridesAndHintsAVX2_i64 : public SortWithStrideFixture<i64> {};
auto vxsort_i64_stride_params_avx2  = ValuesIn(SizeAndStride<i64>::generate(1000000, 0x8L, 0x4000000L, 0x80000000L));
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortWithStridesAndHintsAVX2_i64, vxsort_i64_stride_params_avx2, PrintSizeAndStride<i64>());

TEST_P(VxSortWithStridesAndHintsAVX2_i64, VxSortStridesAndHintsAVX2_1)  { vxsort_hinted_test<i64,  1, 3, VM::AVX2>(V, MinValue, MaxValue); }
TEST_P(VxSortWithStridesAndHintsAVX2_i64, VxSortStridesAndHintsAVX2_2)  { vxsort_hinted_test<i64,  2, 3, VM::AVX2>(V, MinValue, MaxValue); }
TEST_P(VxSortWithStridesAndHintsAVX2_i64, VxSortStridesAndHintsAVX2_4)  { vxsort_hinted_test<i64,  4, 3, VM::AVX2>(V, MinValue, MaxValue); }
TEST_P(VxSortWithStridesAndHintsAVX2_i64, VxSortStridesAndHintsAVX2_8)  { vxsort_hinted_test<i64,  8, 3, VM::AVX2>(V, MinValue, MaxValue); }
TEST_P(VxSortWithStridesAndHintsAVX2_i64, VxSortStridesAndHintsAVX2_12) { vxsort_hinted_test<i64, 12, 3, VM::AVX2>(V, MinValue, MaxValue); }

}

#include "vxsort_targets_disable.h"
