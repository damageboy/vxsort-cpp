#include "vxsort_targets_enable_avx2.h"

#include "gtest/gtest.h"
#include "../fixtures.h"

#include <smallsort/avx2/bitonic_machine.AVX2.i32.generated.h>
#include <smallsort/bitonic_machine.h>
#include <smallsort/bitonic_sort.h>
#include <vector_machine/machine_traits.avx2.h>

#include "packnsort_test.h"

namespace vxsort_tests {

using testing::Types;
using vxsort::i64;
using vxsort::vector_machine;

struct PackNSortTestAVX2_i64 : public SortWithSlackTest<i64> {};

auto packed_vxsort_i64_params_avx2  = ValuesIn(SizeAndSlack<i64>::generate(10, 1000000, 10, 8, 0x1000, 0x8));

INSTANTIATE_TEST_SUITE_P(FullSort, PackNSortTestAVX2_i64, packed_vxsort_i64_params_avx2, PrintSizeAndSlack<i64>());

TEST_P(PackNSortTestAVX2_i64, VxSortAVX2_1)  { perform_packedvxsort_test< 1, vector_machine::AVX2, 3>(V); }
TEST_P(PackNSortTestAVX2_i64, VxSortAVX2_2)  { perform_packedvxsort_test< 2, vector_machine::AVX2, 3>(V); }
TEST_P(PackNSortTestAVX2_i64, VxSortAVX2_4)  { perform_packedvxsort_test< 4, vector_machine::AVX2, 3>(V); }
TEST_P(PackNSortTestAVX2_i64, VxSortAVX2_8)  { perform_packedvxsort_test< 8, vector_machine::AVX2, 3>(V); }
TEST_P(PackNSortTestAVX2_i64, VxSortAVX2_12) { perform_packedvxsort_test<12, vector_machine::AVX2, 3>(V); }

}

#include "vxsort_targets_disable.h"
