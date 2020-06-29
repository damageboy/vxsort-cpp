#include "vxsort_targets_enable_avx2.h"

#include "gtest/gtest.h"
#include "../fixtures.h"

#include <machine_traits.avx2.h>
#include <smallsort/bitonic_sort.AVX2.int32_t.generated.h>

#include "packnsort_test.h"

namespace vxsort_tests {
using testing::Types;

using vxsort::vector_machine;

struct PackNSortTestAVX2_i64 : public SortWithSlackTest<int64_t> {};

auto packed_vxsort_int64_params_avx2  = ValuesIn(SizeAndSlack<int64_t>::generate(10, 1000000, 10, 8, 0x1000, 0x8));

INSTANTIATE_TEST_SUITE_P(FullSort, PackNSortTestAVX2_i64,    packed_vxsort_int64_params_avx2, PrintSizeAndSlack<int64_t>());

TEST_P(PackNSortTestAVX2_i64, VxSortAVX2_1)  { perform_packedvxsort_test< 1, vector_machine::AVX2, 3>(V); }
TEST_P(PackNSortTestAVX2_i64, VxSortAVX2_2)  { perform_packedvxsort_test< 2, vector_machine::AVX2, 3>(V); }
TEST_P(PackNSortTestAVX2_i64, VxSortAVX2_4)  { perform_packedvxsort_test< 4, vector_machine::AVX2, 3>(V); }
TEST_P(PackNSortTestAVX2_i64, VxSortAVX2_8)  { perform_packedvxsort_test< 8, vector_machine::AVX2, 3>(V); }
TEST_P(PackNSortTestAVX2_i64, VxSortAVX2_12) { perform_packedvxsort_test<12, vector_machine::AVX2, 3>(V); }

}

#include "vxsort_targets_disable.h"
