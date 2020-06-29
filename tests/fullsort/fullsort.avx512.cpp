#include "vxsort_targets_enable_avx512.h"

#include "gtest/gtest.h"
#include "../fixtures.h"

#include "fullsort_test.h"
#include <machine_traits.avx512.h>
#include <smallsort/bitonic_sort.AVX512.double.generated.h>
#include <smallsort/bitonic_sort.AVX512.float.generated.h>
#include <smallsort/bitonic_sort.AVX512.int32_t.generated.h>
#include <smallsort/bitonic_sort.AVX512.int64_t.generated.h>
#include <smallsort/bitonic_sort.AVX512.uint32_t.generated.h>
#include <smallsort/bitonic_sort.AVX512.uint64_t.generated.h>

namespace vxsort_tests {
using testing::Types;

using vxsort::vector_machine;

struct FullSortTestAVX512_i32 : public SortWithSlackTest<int32_t> {};
struct FullSortTestAVX512_ui32 : public SortWithSlackTest<uint32_t> {};
struct FullSortTestAVX512_i64 : public SortWithSlackTest<int64_t> {};
struct FullSortTestAVX512_ui64 : public SortWithSlackTest<uint64_t> {};
struct FullSortTestAVX512_float : public SortWithSlackTest<float> {};
struct FullSortTestAVX512_double : public SortWithSlackTest<double> {};

auto vxsort_int32_params_avx512 = ValuesIn(SizeAndSlack<int32_t>::generate(10, 1000000, 10, 32, 0x1000, 0x1));
auto vxsort_uint32_params_avx512 = ValuesIn(SizeAndSlack<uint32_t>::generate(10, 1000000, 10, 32, 0x1000, 0x1));
auto vxsort_float_params_avx512 = ValuesIn(SizeAndSlack<float>::generate(10, 1000000, 10, 32, 1234.5, 0.1));
auto vxsort_int64_params_avx512 = ValuesIn(SizeAndSlack<int64_t>::generate(10, 1000000, 10, 16, 0x1000, 0x1));
auto vxsort_uint64_params_avx512 = ValuesIn(SizeAndSlack<uint64_t>::generate(10, 1000000, 10, 16, 0x1000, 0x1));
auto vxsort_double_params_avx512 = ValuesIn(SizeAndSlack<double>::generate(10, 1000000, 10, 16, 0x1000, 0x1));

INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_i32,    vxsort_int32_params_avx512, PrintSizeAndSlack<int32_t>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_ui32,   vxsort_uint32_params_avx512, PrintSizeAndSlack<uint32_t>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_float,  vxsort_float_params_avx512, PrintSizeAndSlack<float>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_i64,    vxsort_int64_params_avx512, PrintSizeAndSlack<int64_t>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_ui64,   vxsort_uint64_params_avx512, PrintSizeAndSlack<uint64_t>());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX512_double, vxsort_double_params_avx512, PrintSizeAndSlack<double>());

TEST_P(FullSortTestAVX512_i32, VxSortAVX512_1)  { perform_vxsort_test<int32_t,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i32, VxSortAVX512_2)  { perform_vxsort_test<int32_t,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i32, VxSortAVX512_4)  { perform_vxsort_test<int32_t,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i32, VxSortAVX512_8)  { perform_vxsort_test<int32_t,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i32, VxSortAVX512_12) { perform_vxsort_test<int32_t, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_ui32, VxSortAVX512_1)  { perform_vxsort_test<uint32_t,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_ui32, VxSortAVX512_2)  { perform_vxsort_test<uint32_t,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_ui32, VxSortAVX512_4)  { perform_vxsort_test<uint32_t,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_ui32, VxSortAVX512_8)  { perform_vxsort_test<uint32_t,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_ui32, VxSortAVX512_12) { perform_vxsort_test<uint32_t, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_float, VxSortAVX512_1)  { perform_vxsort_test<float,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_float, VxSortAVX512_2)  { perform_vxsort_test<float,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_float, VxSortAVX512_4)  { perform_vxsort_test<float,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_float, VxSortAVX512_8)  { perform_vxsort_test<float,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_float, VxSortAVX512_12) { perform_vxsort_test<float, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_i64, VxSortAVX512_1)  { perform_vxsort_test<int64_t,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i64, VxSortAVX512_2)  { perform_vxsort_test<int64_t,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i64, VxSortAVX512_4)  { perform_vxsort_test<int64_t,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i64, VxSortAVX512_8)  { perform_vxsort_test<int64_t,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_i64, VxSortAVX512_12) { perform_vxsort_test<int64_t, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_ui64, VxSortAVX512_1)  { perform_vxsort_test<uint64_t,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_ui64, VxSortAVX512_2)  { perform_vxsort_test<uint64_t,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_ui64, VxSortAVX512_4)  { perform_vxsort_test<uint64_t,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_ui64, VxSortAVX512_8)  { perform_vxsort_test<uint64_t,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_ui64, VxSortAVX512_12) { perform_vxsort_test<uint64_t, 12, vector_machine::AVX512>(V); }

TEST_P(FullSortTestAVX512_double, VxSortAVX512_1)  { perform_vxsort_test<double,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_double, VxSortAVX512_2)  { perform_vxsort_test<double,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_double, VxSortAVX512_4)  { perform_vxsort_test<double,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_double, VxSortAVX512_8)  { perform_vxsort_test<double,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTestAVX512_double, VxSortAVX512_12) { perform_vxsort_test<double, 12, vector_machine::AVX512>(V); }

}

#include "vxsort_targets_disable.h"
