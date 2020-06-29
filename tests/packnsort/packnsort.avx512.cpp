#include "vxsort_targets_enable_avx512.h"

#include "gtest/gtest.h"
#include "../fixtures.h"

#include "packnsort_test.h"
#include <machine_traits.avx512.h>
#include <smallsort/bitonic_sort.AVX512.int32_t.generated.h>

namespace vxsort_tests {
using testing::Types;

using vxsort::vector_machine;

struct PackNSortTestAVX512_i64 : public SortWithSlackTest<int64_t> {};

auto packed_vxsort_int64_params_avx512  = ValuesIn(SizeAndSlack<int64_t>::generate(10, 1000000, 10, 8, 0x1000, 0x8));

INSTANTIATE_TEST_SUITE_P(FullSort, PackNSortTestAVX512_i64, packed_vxsort_int64_params_avx512, PrintSizeAndSlack<int64_t>());

TEST_P(PackNSortTestAVX512_i64, VxSortAVX512_1)  { perform_packedvxsort_test< 1, vector_machine::AVX512, 3>(V); }
TEST_P(PackNSortTestAVX512_i64, VxSortAVX512_2)  { perform_packedvxsort_test< 2, vector_machine::AVX512, 3>(V); }
TEST_P(PackNSortTestAVX512_i64, VxSortAVX512_4)  { perform_packedvxsort_test< 4, vector_machine::AVX512, 3>(V); }
TEST_P(PackNSortTestAVX512_i64, VxSortAVX512_8)  { perform_packedvxsort_test< 8, vector_machine::AVX512, 3>(V); }
TEST_P(PackNSortTestAVX512_i64, VxSortAVX512_12) { perform_packedvxsort_test<12, vector_machine::AVX512, 3>(V); }

}

#include "vxsort_targets_disable.h"
