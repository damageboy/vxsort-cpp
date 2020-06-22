#include "gtest/gtest.h"
#include "../fixtures.h"

  #include "vxsort_targets_enable_avx2.h"

#include "fullsort_test.h"
#include "machine_traits.avx2.h"
#include "smallsort/bitonic_sort.AVX2.int32_t.generated.h"
#include "smallsort/bitonic_sort.AVX2.uint32_t.generated.h"
#include "smallsort/bitonic_sort.AVX2.float.generated.h"
#include "smallsort/bitonic_sort.AVX2.int64_t.generated.h"
#include "smallsort/bitonic_sort.AVX2.uint64_t.generated.h"
#include "smallsort/bitonic_sort.AVX2.double.generated.h"


namespace vxsort_tests {
using testing::Types;

using vxsort::vector_machine;

struct FullSortTestAVX2_i32 : public SortWithSlackTest<int32_t> {};
struct FullSortTestAVX2_ui32 : public SortWithSlackTest<uint32_t> {};
struct FullSortTestAVX2_i64 : public SortWithSlackTest<int64_t> {};
struct FullSortTestAVX2_ui64 : public SortWithSlackTest<uint64_t> {};
struct FullSortTestAVX2_float : public SortWithSlackTest<float> {};
struct FullSortTestAVX2_double : public SortWithSlackTest<double> {};

auto vxsort_values_avx2_32 = ValuesIn(SizeAndSlack::generate(10, 1000000, 10, 16));
auto vxsort_values_avx2_64 = ValuesIn(SizeAndSlack::generate(10, 1000000, 10, 8));

INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_i32,    vxsort_values_avx2_32, PrintSizeAndSlack());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_ui32,   vxsort_values_avx2_32, PrintSizeAndSlack());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_float,  vxsort_values_avx2_64, PrintSizeAndSlack());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_i64,    vxsort_values_avx2_64, PrintSizeAndSlack());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_ui64,   vxsort_values_avx2_64, PrintSizeAndSlack());
INSTANTIATE_TEST_SUITE_P(FullSort, FullSortTestAVX2_double, vxsort_values_avx2_64, PrintSizeAndSlack());

TEST_P(FullSortTestAVX2_i32, VxSortAVX2_1)  { perform_vxsort_test<int32_t,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i32, VxSortAVX2_2)  { perform_vxsort_test<int32_t,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i32, VxSortAVX2_4)  { perform_vxsort_test<int32_t,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i32, VxSortAVX2_8)  { perform_vxsort_test<int32_t,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i32, VxSortAVX2_12) { perform_vxsort_test<int32_t, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTestAVX2_ui32, VxSortAVX2_1)  { perform_vxsort_test<uint32_t,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_ui32, VxSortAVX2_2)  { perform_vxsort_test<uint32_t,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_ui32, VxSortAVX2_4)  { perform_vxsort_test<uint32_t,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_ui32, VxSortAVX2_8)  { perform_vxsort_test<uint32_t,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_ui32, VxSortAVX2_12) { perform_vxsort_test<uint32_t, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTestAVX2_float, VxSortAVX2_1)  { perform_vxsort_test<float,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_float, VxSortAVX2_2)  { perform_vxsort_test<float,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_float, VxSortAVX2_4)  { perform_vxsort_test<float,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_float, VxSortAVX2_8)  { perform_vxsort_test<float,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_float, VxSortAVX2_12) { perform_vxsort_test<float, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTestAVX2_i64, VxSortAVX2_1)  { perform_vxsort_test<int64_t,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i64, VxSortAVX2_2)  { perform_vxsort_test<int64_t,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i64, VxSortAVX2_4)  { perform_vxsort_test<int64_t,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i64, VxSortAVX2_8)  { perform_vxsort_test<int64_t,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_i64, VxSortAVX2_12) { perform_vxsort_test<int64_t, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTestAVX2_ui64, VxSortAVX2_1)  { perform_vxsort_test<uint64_t,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_ui64, VxSortAVX2_2)  { perform_vxsort_test<uint64_t,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_ui64, VxSortAVX2_4)  { perform_vxsort_test<uint64_t,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_ui64, VxSortAVX2_8)  { perform_vxsort_test<uint64_t,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_ui64, VxSortAVX2_12) { perform_vxsort_test<uint64_t, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTestAVX2_double, VxSortAVX2_1)  { perform_vxsort_test<double,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_double, VxSortAVX2_2)  { perform_vxsort_test<double,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_double, VxSortAVX2_4)  { perform_vxsort_test<double,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_double, VxSortAVX2_8)  { perform_vxsort_test<double,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTestAVX2_double, VxSortAVX2_12) { perform_vxsort_test<double, 12, vector_machine::AVX2>(V); }

}

#include "vxsort_targets_disable.h"
