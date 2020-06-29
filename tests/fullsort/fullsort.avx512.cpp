#include "vxsort_targets_enable_avx512.h"

#include "gtest/gtest.h"

#include <smallsort/avx512/bitonic_machine.avx512.h>
#include <vector_machine/machine_traits.avx512.h>

#include "fullsort_test.h"
#include "../fixtures.h"

namespace vxsort_tests {
using namespace vxsort::types;
using testing::Types;

using namespace vxsort;

struct FullSortTestAVX512_i16 : public SortWithSlackTest<i16> {};
struct FullSortTestAVX512_i32 : public SortWithSlackTest<i32> {};
struct FullSortTestAVX512_i64 : public SortWithSlackTest<i64> {};
struct FullSortTestAVX512_u16 : public SortWithSlackTest<u16> {};
struct FullSortTestAVX512_u32 : public SortWithSlackTest<u32> {};
struct FullSortTestAVX512_u64 : public SortWithSlackTest<u64> {};
struct FullSortTestAVX512_f32 : public SortWithSlackTest<f32> {};
struct FullSortTestAVX512_f64 : public SortWithSlackTest<f64> {};

auto vxsort_i16_params_avx512 = ValuesIn(SizeAndSlack<i16>::generate(10, 10000,   10, 32, 0x1000, 0x1));
auto vxsort_i32_params_avx512 = ValuesIn(SizeAndSlack<i32>::generate(10, 1000000, 10, 32, 0x1000, 0x1));
auto vxsort_i64_params_avx512 = ValuesIn(SizeAndSlack<i64>::generate(10, 1000000, 10, 16, 0x1000, 0x1));
auto vxsort_u16_params_avx512 = ValuesIn(SizeAndSlack<u16>::generate(10, 10000,   10, 32, 0x1000, 0x1));
auto vxsort_u32_params_avx512 = ValuesIn(SizeAndSlack<u32>::generate(10, 1000000, 10, 32, 0x1000, 0x1));
auto vxsort_u64_params_avx512 = ValuesIn(SizeAndSlack<u64>::generate(10, 1000000, 10, 16, 0x1000, 0x1));
auto vxsort_f32_params_avx512 = ValuesIn(SizeAndSlack<f32>::generate(10, 1000000, 10, 32, 1234.5, 0.1f));
auto vxsort_f64_params_avx512 = ValuesIn(SizeAndSlack<f64>::generate(10, 1000000, 10, 16, 0x1000, 0x1));

INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_i16, vxsort_i16_params_avx512, PrintSizeAndSlack<i16>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_i32, vxsort_i32_params_avx512, PrintSizeAndSlack<i32>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_i64, vxsort_i64_params_avx512, PrintSizeAndSlack<i64>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_u16, vxsort_u16_params_avx512, PrintSizeAndSlack<u16>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_u32, vxsort_u32_params_avx512, PrintSizeAndSlack<u32>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_u64, vxsort_u64_params_avx512, PrintSizeAndSlack<u64>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_f32, vxsort_f32_params_avx512, PrintSizeAndSlack<f32>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_f64, vxsort_f64_params_avx512, PrintSizeAndSlack<f64>());

TEST_P(FullSortTestAVX512_i16, VxSortAVX512_1)  { vxsort_test<i16,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i16, VxSortAVX512_2)  { vxsort_test<i16,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i16, VxSortAVX512_4)  { vxsort_test<i16,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i16, VxSortAVX512_8)  { vxsort_test<i16,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i16, VxSortAVX512_12) { vxsort_test<i16, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_i32, VxSortAVX512_1)  { vxsort_test<i32,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i32, VxSortAVX512_2)  { vxsort_test<i32,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i32, VxSortAVX512_4)  { vxsort_test<i32,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i32, VxSortAVX512_8)  { vxsort_test<i32,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i32, VxSortAVX512_12) { vxsort_test<i32, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_i64, VxSortAVX512_1)  { vxsort_test<i64,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i64, VxSortAVX512_2)  { vxsort_test<i64,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i64, VxSortAVX512_4)  { vxsort_test<i64,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i64, VxSortAVX512_8)  { vxsort_test<i64,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i64, VxSortAVX512_12) { vxsort_test<i64, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_u16, VxSortAVX512_1)  { vxsort_test<u16,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_u16, VxSortAVX512_2)  { vxsort_test<u16,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_u16, VxSortAVX512_4)  { vxsort_test<u16,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_u16, VxSortAVX512_8)  { vxsort_test<u16,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_u16, VxSortAVX512_12) { vxsort_test<u16, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_u32, VxSortAVX512_1)  { vxsort_test<u32,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_u32, VxSortAVX512_2)  { vxsort_test<u32,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_u32, VxSortAVX512_4)  { vxsort_test<u32,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_u32, VxSortAVX512_8)  { vxsort_test<u32,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_u32, VxSortAVX512_12) { vxsort_test<u32, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_u64, VxSortAVX512_1)  { vxsort_test<u64,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_u64, VxSortAVX512_2)  { vxsort_test<u64,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_u64, VxSortAVX512_4)  { vxsort_test<u64,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_u64, VxSortAVX512_8)  { vxsort_test<u64,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_u64, VxSortAVX512_12) { vxsort_test<u64, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_f32, VxSortAVX512_1)  { vxsort_test<f32,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_f32, VxSortAVX512_2)  { vxsort_test<f32,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_f32, VxSortAVX512_4)  { vxsort_test<f32,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_f32, VxSortAVX512_8)  { vxsort_test<f32,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_f32, VxSortAVX512_12) { vxsort_test<f32, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_f64, VxSortAVX512_1)  { vxsort_test<f64,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_f64, VxSortAVX512_2)  { vxsort_test<f64,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_f64, VxSortAVX512_4)  { vxsort_test<f64,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_f64, VxSortAVX512_8)  { vxsort_test<f64,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_f64, VxSortAVX512_12) { vxsort_test<f64, 12, vector_machine::AVX512>(V); }

struct PackingStridedSortTestAVX512_i64 : public SortWithStrideTest<i64> {};
auto vxsort_int64_stride_params_avx512  = ValuesIn(SizeAndStride<i64>::generate(1000000, 0x8L, 0x1000000L, 0x80000000L));
INSTANTIATE_TEST_SUITE_P(FullPackingSort, PackingStridedSortTestAVX512_i64, vxsort_int64_stride_params_avx512, PrintSizeAndStride<i64>());

//TEST_P(PackingStridedSortTestAVX512_i64, VxSortStridedAVX512_1)  { vxsort_hinted_test<i64,  1, 3, vector_machine::AVX512>(V, MinValue, MaxValue); }
//TEST_P(PackingStridedSortTestAVX512_i64, VxSortStridedAVX512_2)  { vxsort_hinted_test<i64,  2, 3, vector_machine::AVX512>(V, MinValue, MaxValue); }
//TEST_P(PackingStridedSortTestAVX512_i64, VxSortStridedAVX512_4)  { vxsort_hinted_test<i64,  4, 3, vector_machine::AVX512>(V, MinValue, MaxValue); }
TEST_P(PackingStridedSortTestAVX512_i64, VxSortStridedAVX512_8)  { vxsort_hinted_test<i64,  8, 3, vector_machine::AVX512>(V, MinValue, MaxValue); }
//TEST_P(PackingStridedSortTestAVX512_i64, VxSortStridedAVX512_12) { vxsort_hinted_test<i64, 12, 3, vector_machine::AVX512>(V, MinValue, MaxValue); }

}

#include "vxsort_targets_disable.h"
