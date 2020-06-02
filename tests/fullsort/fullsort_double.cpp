#include "gtest/gtest.h"
#include "../fixtures.h"
#include "fullsort.h"

#include "vxsort_targets_enable.h"

#include <vxsort.double.h>

namespace gcsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using gcsort::vector_machine;

struct FullSortTest_double : public SortWithSlackTest<double> {};

INSTANTIATE_TEST_SUITE_P(FullSortSizes,
                         FullSortTest_double,
                         ValuesIn(SizeAndSlack::generate(10, 1000000, 10, FullSortTest_double::VectorElements*2)),
                         PrintSizeAndSlack());
TEST_P(FullSortTest_double, VxSortAVX2_1)  { perform_vxsort_test<double,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_double, VxSortAVX2_2)  { perform_vxsort_test<double,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_double, VxSortAVX2_4)  { perform_vxsort_test<double,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_double, VxSortAVX2_8)  { perform_vxsort_test<double,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_double, VxSortAVX2_12) { perform_vxsort_test<double, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTest_double, VxSortAVX512_1)  { perform_vxsort_test<double,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_double, VxSortAVX512_2)  { perform_vxsort_test<double,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_double, VxSortAVX512_4)  { perform_vxsort_test<double,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_double, VxSortAVX512_8)  { perform_vxsort_test<double,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_double, VxSortAVX512_12) { perform_vxsort_test<double, 12, vector_machine::AVX512>(V); }

}

#include "vxsort_targets_disable.h"