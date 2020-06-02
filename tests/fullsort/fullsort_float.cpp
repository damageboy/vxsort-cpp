#include "gtest/gtest.h"
#include "../fixtures.h"
#include "fullsort.h"

#include "vxsort_targets_enable.h"

#include <vxsort.float.h>


namespace gcsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using gcsort::vector_machine;

struct FullSortTest_float : public SortWithSlackTest<float> {};


INSTANTIATE_TEST_SUITE_P(FullSortSizes,
                         FullSortTest_float,
                         ValuesIn(SizeAndSlack::generate(10, 1000000, 10, FullSortTest_float::VectorElements*2)),
                         PrintSizeAndSlack());

TEST_P(FullSortTest_float, VxSortAVX2_1)  { perform_vxsort_test<float,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_float, VxSortAVX2_2)  { perform_vxsort_test<float,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_float, VxSortAVX2_4)  { perform_vxsort_test<float,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_float, VxSortAVX2_8)  { perform_vxsort_test<float,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_float, VxSortAVX2_12) { perform_vxsort_test<float, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTest_float, VxSortAVX512_1)  { perform_vxsort_test<float,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_float, VxSortAVX512_2)  { perform_vxsort_test<float,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_float, VxSortAVX512_4)  { perform_vxsort_test<float,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_float, VxSortAVX512_8)  { perform_vxsort_test<float,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_float, VxSortAVX512_12) { perform_vxsort_test<float, 12, vector_machine::AVX512>(V); }

}

#include "vxsort_targets_disable.h"