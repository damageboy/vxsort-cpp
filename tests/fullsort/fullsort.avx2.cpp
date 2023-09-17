#include "vxsort_targets_enable_avx2.h"

#include "gtest/gtest.h"

#include <vxsort.avx2.h>
#include "fullsort_test.h"
#include "../sort_fixtures.h"

namespace vxsort_tests {
using namespace vxsort::types;
using testing::Types;

using VM = vxsort::vector_machine;
using namespace vxsort;

#ifdef VXSORT_TEST_AVX2_I16
struct VxSortAVX2_i16 : public ParametrizedSortFixture<i16> {};
auto vxsort_i16_params_avx2 = ValuesIn(SortTestParams<i16>::gen_mult(SortPattern::unique_values, 10, 10000,   10, 32, 0x1000, 0x1));
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortAVX2_i16, vxsort_i16_params_avx2, PrintSortTestParams<i16>());
#endif
#ifdef VXSORT_TEST_AVX2_I32
struct VxSortAVX2_i32 : public ParametrizedSortFixture<i32> {};
auto vxsort_i32_params_avx2 = ValuesIn(SortTestParams<i32>::gen_mult(SortPattern::unique_values, 10, 1000000, 10, 16, 0x1000, 0x1));
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortAVX2_i32, vxsort_i32_params_avx2, PrintSortTestParams<i32>());
#endif
#ifdef VXSORT_TEST_AVX2_I64
struct VxSortAVX2_i64 : public ParametrizedSortFixture<i64> {};
auto vxsort_i64_params_avx2 = ValuesIn(SortTestParams<i64>::gen_mult(SortPattern::unique_values, 10, 1000000, 10, 8, 0x1000, 0x1));
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortAVX2_i64, vxsort_i64_params_avx2, PrintSortTestParams<i64>());
#endif
#ifdef VXSORT_TEST_AVX2_U16
struct VxSortAVX2_u16 : public ParametrizedSortFixture<u16> {};
auto vxsort_u16_params_avx2 = ValuesIn(SortTestParams<u16>::gen_mult(SortPattern::unique_values, 10, 1000000, 10, 16, 0x1000, 0x1));
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortAVX2_u16, vxsort_u16_params_avx2, PrintSortTestParams<u16>());
#endif
#ifdef VXSORT_TEST_AVX2_U32
struct VxSortAVX2_u32 : public ParametrizedSortFixture<u32> {};
auto vxsort_u32_params_avx2 = ValuesIn(SortTestParams<u32>::gen_mult(SortPattern::unique_values, 10, 1000000, 10, 16, 0x1000, 0x1));
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortAVX2_u32, vxsort_u32_params_avx2, PrintSortTestParams<u32>());
#endif
#ifdef VXSORT_TEST_AVX2_U64
struct VxSortAVX2_u64 : public ParametrizedSortFixture<u64> {};
auto vxsort_u64_params_avx2 = ValuesIn(SortTestParams<u64>::gen_mult(SortPattern::unique_values, 10, 1000000, 10, 8, 0x1000, 0x1));
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortAVX2_u64, vxsort_u64_params_avx2, PrintSortTestParams<u64>());
#endif
#ifdef VXSORT_TEST_AVX2_F32
struct VxSortAVX2_f32 : public ParametrizedSortFixture<f32> {};
auto vxsort_f32_params_avx2 = ValuesIn(SortTestParams<f32>::gen_mult(SortPattern::unique_values, 10, 1000000, 10, 16, 1234.5f, 0.1f));
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortAVX2_f32, vxsort_f32_params_avx2, PrintSortTestParams<f32>());
#endif
#ifdef VXSORT_TEST_AVX2_F64
struct VxSortAVX2_f64 : public ParametrizedSortFixture<f64> {};
auto vxsort_f64_params_avx2 = ValuesIn(SortTestParams<f64>::gen_mult(SortPattern::unique_values, 10, 1000000, 10, 8, 1234.5, 0.1));
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortAVX2_f64, vxsort_f64_params_avx2, PrintSortTestParams<f64>());
#endif

#ifdef VXSORT_TEST_AVX2_I16
TEST_P(VxSortAVX2_i16, VxSortAVX2_1)  { vxsort_test<i16,  1, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i16, VxSortAVX2_2)  { vxsort_test<i16,  2, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i16, VxSortAVX2_4)  { vxsort_test<i16,  4, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i16, VxSortAVX2_8)  { vxsort_test<i16,  8, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i16, VxSortAVX2_12) { vxsort_test<i16, 12, VM::AVX2>(V); }
#endif

#ifdef VXSORT_TEST_AVX2_I32
TEST_P(VxSortAVX2_i32, VxSortAVX2_1)  { vxsort_test<i32,  1, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i32, VxSortAVX2_2)  { vxsort_test<i32,  2, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i32, VxSortAVX2_4)  { vxsort_test<i32,  4, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i32, VxSortAVX2_8)  { vxsort_test<i32,  8, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i32, VxSortAVX2_12) { vxsort_test<i32, 12, VM::AVX2>(V); }
#endif

#ifdef VXSORT_TEST_AVX2_U16
TEST_P(VxSortAVX2_u16, VxSortAVX2_1)  { vxsort_test<u16,  1, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u16, VxSortAVX2_2)  { vxsort_test<u16,  2, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u16, VxSortAVX2_4)  { vxsort_test<u16,  4, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u16, VxSortAVX2_8)  { vxsort_test<u16,  8, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u16, VxSortAVX2_12) { vxsort_test<u16, 12, VM::AVX2>(V); }
#endif

#ifdef VXSORT_TEST_AVX2_U32
TEST_P(VxSortAVX2_u32, VxSortAVX2_1)  { vxsort_test<u32,  1, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u32, VxSortAVX2_2)  { vxsort_test<u32,  2, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u32, VxSortAVX2_4)  { vxsort_test<u32,  4, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u32, VxSortAVX2_8)  { vxsort_test<u32,  8, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u32, VxSortAVX2_12) { vxsort_test<u32, 12, VM::AVX2>(V); }
#endif

#ifdef VXSORT_TEST_AVX2_F32
TEST_P(VxSortAVX2_f32, VxSortAVX2_1)  { vxsort_test<f32,  1, VM::AVX2>(V); }
TEST_P(VxSortAVX2_f32, VxSortAVX2_2)  { vxsort_test<f32,  2, VM::AVX2>(V); }
TEST_P(VxSortAVX2_f32, VxSortAVX2_4)  { vxsort_test<f32,  4, VM::AVX2>(V); }
TEST_P(VxSortAVX2_f32, VxSortAVX2_8)  { vxsort_test<f32,  8, VM::AVX2>(V); }
TEST_P(VxSortAVX2_f32, VxSortAVX2_12) { vxsort_test<f32, 12, VM::AVX2>(V); }
#endif

#ifdef VXSORT_TEST_AVX2_I64
TEST_P(VxSortAVX2_i64, VxSortAVX2_1)  { vxsort_test<i64,  1, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i64, VxSortAVX2_2)  { vxsort_test<i64,  2, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i64, VxSortAVX2_4)  { vxsort_test<i64,  4, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i64, VxSortAVX2_8)  { vxsort_test<i64,  8, VM::AVX2>(V); }
TEST_P(VxSortAVX2_i64, VxSortAVX2_12) { vxsort_test<i64, 12, VM::AVX2>(V); }
#endif

#ifdef VXSORT_TEST_AVX2_U64
TEST_P(VxSortAVX2_u64, VxSortAVX2_1)  { vxsort_test<u64,  1, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u64, VxSortAVX2_2)  { vxsort_test<u64,  2, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u64, VxSortAVX2_4)  { vxsort_test<u64,  4, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u64, VxSortAVX2_8)  { vxsort_test<u64,  8, VM::AVX2>(V); }
TEST_P(VxSortAVX2_u64, VxSortAVX2_12) { vxsort_test<u64, 12, VM::AVX2>(V); }
#endif

#ifdef VXSORT_TEST_AVX2_F64
TEST_P(VxSortAVX2_f64, VxSortAVX2_1)  { vxsort_test<f64,  1, VM::AVX2>(V); }
TEST_P(VxSortAVX2_f64, VxSortAVX2_2)  { vxsort_test<f64,  2, VM::AVX2>(V); }
TEST_P(VxSortAVX2_f64, VxSortAVX2_4)  { vxsort_test<f64,  4, VM::AVX2>(V); }
TEST_P(VxSortAVX2_f64, VxSortAVX2_8)  { vxsort_test<f64,  8, VM::AVX2>(V); }
TEST_P(VxSortAVX2_f64, VxSortAVX2_12) { vxsort_test<f64, 12, VM::AVX2>(V); }
#endif

/*
struct VxSortWithStridesAndHintsAVX2_i64 : public SortWithStrideFixture<i64> {};
auto vxsort_i64_stride_params_avx2  = ValuesIn(SizeAndStride<i64>::generate(1000000, 0x8L, 0x4000000L, 0x80000000L));
INSTANTIATE_TEST_SUITE_P(VxSort, VxSortWithStridesAndHintsAVX2_i64, vxsort_i64_stride_params_avx2, PrintSizeAndStride<i64>());

TEST_P(VxSortWithStridesAndHintsAVX2_i64, VxSortStridesAndHintsAVX2_1)  { vxsort_hinted_test<i64,  1, 3, VM::AVX2>(V, MinValue, MaxValue); }
TEST_P(VxSortWithStridesAndHintsAVX2_i64, VxSortStridesAndHintsAVX2_2)  { vxsort_hinted_test<i64,  2, 3, VM::AVX2>(V, MinValue, MaxValue); }
TEST_P(VxSortWithStridesAndHintsAVX2_i64, VxSortStridesAndHintsAVX2_4)  { vxsort_hinted_test<i64,  4, 3, VM::AVX2>(V, MinValue, MaxValue); }
TEST_P(VxSortWithStridesAndHintsAVX2_i64, VxSortStridesAndHintsAVX2_8)  { vxsort_hinted_test<i64,  8, 3, VM::AVX2>(V, MinValue, MaxValue); }
TEST_P(VxSortWithStridesAndHintsAVX2_i64, VxSortStridesAndHintsAVX2_12) { vxsort_hinted_test<i64, 12, 3, VM::AVX2>(V, MinValue, MaxValue); }
*/
}

#include "vxsort_targets_disable.h"
