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
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using vxsort::vector_machine;

struct FullSortTest_i32 : public SortWithSlackTest<int32_t> {};
struct FullSortTest_ui32 : public SortWithSlackTest<uint32_t> {};
struct FullSortTest_i64 : public SortWithSlackTest<int64_t> {};
struct FullSortTest_ui64 : public SortWithSlackTest<uint64_t> {};
struct FullSortTest_float : public SortWithSlackTest<float> {};
struct FullSortTest_double : public SortWithSlackTest<double> {};

INSTANTIATE_TEST_SUITE_P(FullSortSizes,
                         FullSortTest_i32,
                         ValuesIn(SizeAndSlack::generate(10, 1000000, 10, FullSortTest_i32::VectorElements*2)),
                         //ValuesIn(SizeAndSlack::generate(10, 1000, 10, FullSortTest_i32::VectorElements*2)),
                         PrintSizeAndSlack());

INSTANTIATE_TEST_SUITE_P(FullSortSizes,
                         FullSortTest_ui32,
                         ValuesIn(SizeAndSlack::generate(10, 1000000, 10, FullSortTest_ui32::VectorElements*2)),
                         PrintSizeAndSlack());

INSTANTIATE_TEST_SUITE_P(FullSortSizes,
                         FullSortTest_float,
                         ValuesIn(SizeAndSlack::generate(10, 1000000, 10, FullSortTest_float::VectorElements*2)),
                         PrintSizeAndSlack());

INSTANTIATE_TEST_SUITE_P(FullSortSizes,
                         FullSortTest_i64,
                         ValuesIn(SizeAndSlack::generate(10, 1000000, 10, FullSortTest_i64::VectorElements*2)),
                         PrintSizeAndSlack());

INSTANTIATE_TEST_SUITE_P(FullSortSizes,
                         FullSortTest_ui64,
                         ValuesIn(SizeAndSlack::generate(10, 1000000, 10, FullSortTest_ui64::VectorElements*2)),
                         PrintSizeAndSlack());


INSTANTIATE_TEST_SUITE_P(FullSortSizes,
                         FullSortTest_double,
                         ValuesIn(SizeAndSlack::generate(10, 1000000, 10, FullSortTest_double::VectorElements*2)),
                         PrintSizeAndSlack());

TEST_P(FullSortTest_i32, VxSortAVX2_1)  { perform_vxsort_test<int32_t,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_i32, VxSortAVX2_2)  { perform_vxsort_test<int32_t,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_i32, VxSortAVX2_4)  { perform_vxsort_test<int32_t,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_i32, VxSortAVX2_8)  { perform_vxsort_test<int32_t,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_i32, VxSortAVX2_12) { perform_vxsort_test<int32_t, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTest_ui32, VxSortAVX2_1)  { perform_vxsort_test<uint32_t,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_ui32, VxSortAVX2_2)  { perform_vxsort_test<uint32_t,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_ui32, VxSortAVX2_4)  { perform_vxsort_test<uint32_t,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_ui32, VxSortAVX2_8)  { perform_vxsort_test<uint32_t,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_ui32, VxSortAVX2_12) { perform_vxsort_test<uint32_t, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTest_float, VxSortAVX2_1)  { perform_vxsort_test<float,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_float, VxSortAVX2_2)  { perform_vxsort_test<float,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_float, VxSortAVX2_4)  { perform_vxsort_test<float,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_float, VxSortAVX2_8)  { perform_vxsort_test<float,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_float, VxSortAVX2_12) { perform_vxsort_test<float, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTest_i64, VxSortAVX2_1)  { perform_vxsort_test<int64_t,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_i64, VxSortAVX2_2)  { perform_vxsort_test<int64_t,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_i64, VxSortAVX2_4)  { perform_vxsort_test<int64_t,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_i64, VxSortAVX2_8)  { perform_vxsort_test<int64_t,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_i64, VxSortAVX2_12) { perform_vxsort_test<int64_t, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTest_ui64, VxSortAVX2_1)  { perform_vxsort_test<uint64_t,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_ui64, VxSortAVX2_2)  { perform_vxsort_test<uint64_t,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_ui64, VxSortAVX2_4)  { perform_vxsort_test<uint64_t,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_ui64, VxSortAVX2_8)  { perform_vxsort_test<uint64_t,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_ui64, VxSortAVX2_12) { perform_vxsort_test<uint64_t, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTest_double, VxSortAVX2_1)  { perform_vxsort_test<double,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_double, VxSortAVX2_2)  { perform_vxsort_test<double,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_double, VxSortAVX2_4)  { perform_vxsort_test<double,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_double, VxSortAVX2_8)  { perform_vxsort_test<double,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_double, VxSortAVX2_12) { perform_vxsort_test<double, 12, vector_machine::AVX2>(V); }


}

#include "vxsort_targets_disable.h"