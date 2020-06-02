#include "gtest/gtest.h"
#include "../fixtures.h"
#include "fullsort.h"

#include "vxsort_targets_enable.h"

#include <vxsort.int32_t.h>

namespace gcsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using gcsort::vector_machine;

struct FullSortTest_i32 : public SortWithSlackTest<int32_t> {};

INSTANTIATE_TEST_SUITE_P(FullSortSizes,
                         FullSortTest_i32,
                         ValuesIn(SizeAndSlack::generate(10, 1000000, 10, FullSortTest_i32::VectorElements*2)),
                         //ValuesIn(SizeAndSlack::generate(10, 1000, 10, FullSortTest_i32::VectorElements*2)),
                         PrintSizeAndSlack());

TEST_P(FullSortTest_i32, VxSortAVX2_1)  { perform_vxsort_test<int32_t,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_i32, VxSortAVX2_2)  { perform_vxsort_test<int32_t,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_i32, VxSortAVX2_4)  { perform_vxsort_test<int32_t,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_i32, VxSortAVX2_8)  { perform_vxsort_test<int32_t,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_i32, VxSortAVX2_12) { perform_vxsort_test<int32_t, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTest_i32, VxSortAVX512_1)  { perform_vxsort_test<int32_t,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_i32, VxSortAVX512_2)  { perform_vxsort_test<int32_t,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_i32, VxSortAVX512_4)  { perform_vxsort_test<int32_t,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_i32, VxSortAVX512_8)  { perform_vxsort_test<int32_t,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_i32, VxSortAVX512_12) { perform_vxsort_test<int32_t, 12, vector_machine::AVX512>(V); }

}

#include "vxsort_targets_disable.h"