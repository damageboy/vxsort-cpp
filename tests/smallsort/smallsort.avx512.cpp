#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "../fixtures.h"


#include "vxsort_targets_enable_avx512.h"

#include "smallsort_test.h"

#include <smallsort/bitonic_sort.AVX512.int32_t.generated.h>
#include <smallsort/bitonic_sort.AVX512.uint32_t.generated.h>
#include <smallsort/bitonic_sort.AVX512.float.generated.h>
#include <smallsort/bitonic_sort.AVX512.int64_t.generated.h>
#include <smallsort/bitonic_sort.AVX512.uint64_t.generated.h>
#include <smallsort/bitonic_sort.AVX512.double.generated.h>

namespace vxsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using vxsort::vector_machine;

struct BitonicAVX512_i32 : public SortTest<int32_t> {};
struct BitonicAVX512_ui32 : public SortTest<uint32_t> {};
struct BitonicAVX512_float : public SortTest<float> {};
struct BitonicAVX512_i64 : public SortTest<int64_t> {};
struct BitonicAVX512_ui64 : public SortTest<uint64_t> {};
struct BitonicAVX512_double : public SortTest<double> {};


auto values_avx512_32 = ValuesIn(range(16, 256, 16));
auto values_avx512_64 = ValuesIn(range(8, 128, 8));


INSTANTIATE_TEST_SUITE_P(Bitonic, BitonicAVX512_i32,    values_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(Bitonic, BitonicAVX512_ui32,   values_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(Bitonic, BitonicAVX512_float,  values_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(Bitonic, BitonicAVX512_i64,    values_avx512_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(Bitonic, BitonicAVX512_ui64,   values_avx512_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(Bitonic, BitonicAVX512_double, values_avx512_64, PrintValue());

TEST_P(BitonicAVX512_i32,    BitonicSortAVX512) { perform_bitonic_sort_test<int32_t,  vector_machine::AVX512>(V); }
TEST_P(BitonicAVX512_ui32,   BitonicSortAVX512) { perform_bitonic_sort_test<uint32_t, vector_machine::AVX512>(V); }
TEST_P(BitonicAVX512_float,  BitonicSortAVX512) { perform_bitonic_sort_test<float,    vector_machine::AVX512>(V); }
TEST_P(BitonicAVX512_i64,    BitonicSortAVX512) { perform_bitonic_sort_test<int64_t,  vector_machine::AVX512>(V); }
TEST_P(BitonicAVX512_ui64,   BitonicSortAVX512) { perform_bitonic_sort_test<uint64_t, vector_machine::AVX512>(V); }
TEST_P(BitonicAVX512_double, BitonicSortAVX512) { perform_bitonic_sort_test<double,   vector_machine::AVX512>(V); }

}

#include "vxsort_targets_disable.h"