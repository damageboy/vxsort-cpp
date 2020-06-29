#include "vxsort_targets_enable_avx2.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "../fixtures.h"

#include "smallsort_test.h"

#include <smallsort/bitonic_sort.AVX2.int32_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.uint32_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.float.generated.h>
#include <smallsort/bitonic_sort.AVX2.int64_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.uint64_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.double.generated.h>

namespace vxsort_tests {

using vxsort::vector_machine;

struct BitonicAVX2_i32 : public SortTest<int32_t> {};
struct BitonicAVX2_ui32 : public SortTest<uint32_t> {};
struct BitonicAVX2_float : public SortTest<float> {};
struct BitonicAVX2_i64 : public SortTest<int64_t> {};
struct BitonicAVX2_ui64 : public SortTest<uint64_t> {};
struct BitonicAVX2_double : public SortTest<double> {};

auto bitonic_values_avx2_32 = ValuesIn(range(8, 128, 8));
auto bitonic_values_avx2_64 = ValuesIn(range(4, 64, 4));

INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_i32,    bitonic_values_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_ui32,   bitonic_values_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_float,  bitonic_values_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_i64,    bitonic_values_avx2_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_ui64,   bitonic_values_avx2_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_double, bitonic_values_avx2_64, PrintValue());

TEST_P(BitonicAVX2_i32,    BitonicSortAVX2) { perform_bitonic_sort_test<int32_t,  vector_machine::AVX2>(V); }
TEST_P(BitonicAVX2_ui32,   BitonicSortAVX2) { perform_bitonic_sort_test<uint32_t, vector_machine::AVX2>(V); }
TEST_P(BitonicAVX2_float,  BitonicSortAVX2) { perform_bitonic_sort_test<float,    vector_machine::AVX2>(V); }
TEST_P(BitonicAVX2_i64,    BitonicSortAVX2) { perform_bitonic_sort_test<int64_t,  vector_machine::AVX2>(V); }
TEST_P(BitonicAVX2_ui64,   BitonicSortAVX2) { perform_bitonic_sort_test<uint64_t, vector_machine::AVX2>(V); }
TEST_P(BitonicAVX2_double, BitonicSortAVX2) { perform_bitonic_sort_test<double,   vector_machine::AVX2>(V); }

}
#include "vxsort_targets_disable.h"
