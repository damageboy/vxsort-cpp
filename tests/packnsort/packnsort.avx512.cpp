#include "vxsort_targets_enable_avx512.h"

#include "gtest/gtest.h"
#include "../fixtures.h"

#include <smallsort/avx512/bitonic_machine.AVX512.i32.generated.h>
#include <vector_machine/machine_traits.avx512.h>
#include "packnsort_test.h"

namespace vxsort_tests {

using testing::Types;
using vxsort::vector_machine;
using vxsort::i64;

struct PackNSortTestAVX512_i64 : public SortWithSlackTest<i64> {};

auto packed_vxsort_i64_params_avx512  = ValuesIn(SizeAndSlack<i64>::generate(10, 1000000, 10, 8, 0x1000, 0x8));

INSTANTIATE_TEST_SUITE_P(FullSort, PackNSortTestAVX512_i64, packed_vxsort_i64_params_avx512, PrintSizeAndSlack<i64>());

TEST_P(PackNSortTestAVX512_i64, VxSortAVX512_1)  { perform_packedvxsort_test< 1, vector_machine::AVX512, 3>(V); }
TEST_P(PackNSortTestAVX512_i64, VxSortAVX512_2)  { perform_packedvxsort_test< 2, vector_machine::AVX512, 3>(V); }
TEST_P(PackNSortTestAVX512_i64, VxSortAVX512_4)  { perform_packedvxsort_test< 4, vector_machine::AVX512, 3>(V); }
TEST_P(PackNSortTestAVX512_i64, VxSortAVX512_8)  { perform_packedvxsort_test< 8, vector_machine::AVX512, 3>(V); }
TEST_P(PackNSortTestAVX512_i64, VxSortAVX512_12) { perform_packedvxsort_test<12, vector_machine::AVX512, 3>(V); }

}

#include "vxsort_targets_disable.h"
