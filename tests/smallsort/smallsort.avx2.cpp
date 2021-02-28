#include "vxsort_targets_enable_avx2.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "../fixtures.h"

#include "smallsort_test.h"

#include <smallsort/avx2/bitonic_machine.AVX2.double.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.float.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.int16_t.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.int32_t.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.int64_t.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.uint16_t.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.uint32_t.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.uint64_t.generated.h>
#include <vector_machine/machine_traits.avx2.h>

namespace vxsort_tests {

using VM = vxsort::vector_machine;


struct BitonicMachineAVX2_i16    : public SortTest<int16_t> {};
struct BitonicMachineAVX2_ui16   : public SortTest<uint16_t> {};
struct BitonicMachineAVX2_i32    : public SortTest<int32_t> {};
struct BitonicMachineAVX2_ui32   : public SortTest<uint32_t> {};
struct BitonicMachineAVX2_float  : public SortTest<float> {};
struct BitonicMachineAVX2_i64    : public SortTest<int64_t> {};
struct BitonicMachineAVX2_ui64   : public SortTest<uint64_t> {};
struct BitonicMachineAVX2_double : public SortTest<double> {};

auto bitonic_machine_allvalues_avx2_16 = ValuesIn(range(16, 64, 16));
auto bitonic_machine_allvalues_avx2_32 = ValuesIn(range(8, 32, 8));
auto bitonic_machine_allvalues_avx2_64 = ValuesIn(range(4, 16, 4));

INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_i16,    bitonic_machine_allvalues_avx2_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_ui16,   bitonic_machine_allvalues_avx2_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_i32,    bitonic_machine_allvalues_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_ui32,   bitonic_machine_allvalues_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_float,  bitonic_machine_allvalues_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_i64,    bitonic_machine_allvalues_avx2_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_ui64,   bitonic_machine_allvalues_avx2_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_double, bitonic_machine_allvalues_avx2_64, PrintValue());

TEST_P(BitonicMachineAVX2_i16,    BitonicSortAVX2Asc) { perform_bitonic_machine_sort_test<int16_t,  VM::AVX2, true>(V); }
TEST_P(BitonicMachineAVX2_ui16,   BitonicSortAVX2Asc) { perform_bitonic_machine_sort_test<uint16_t, VM::AVX2, true>(V); }
TEST_P(BitonicMachineAVX2_i32,    BitonicSortAVX2Asc) { perform_bitonic_machine_sort_test<int32_t,  VM::AVX2, true>(V); }
TEST_P(BitonicMachineAVX2_ui32,   BitonicSortAVX2Asc) { perform_bitonic_machine_sort_test<uint32_t, VM::AVX2, true>(V); }
TEST_P(BitonicMachineAVX2_float,  BitonicSortAVX2Asc) { perform_bitonic_machine_sort_test<float,    VM::AVX2, true>(V); }
TEST_P(BitonicMachineAVX2_i64,    BitonicSortAVX2Asc) { perform_bitonic_machine_sort_test<int64_t,  VM::AVX2, true>(V); }
TEST_P(BitonicMachineAVX2_ui64,   BitonicSortAVX2Asc) { perform_bitonic_machine_sort_test<uint64_t, VM::AVX2, true>(V); }
TEST_P(BitonicMachineAVX2_double, BitonicSortAVX2Asc) { perform_bitonic_machine_sort_test<double,   VM::AVX2, true>(V); }

//TEST_P(BitonicMachineAVX2_i32,    BitonicSortAVX2Desc) { perform_bitonic_machine_sort_test<int32_t,  VM::AVX2, false>(V); }
//TEST_P(BitonicMachineAVX2_ui32,   BitonicSortAVX2Desc) { perform_bitonic_machine_sort_test<uint32_t, VM::AVX2, false>(V); }
//TEST_P(BitonicMachineAVX2_float,  BitonicSortAVX2Desc) { perform_bitonic_machine_sort_test<float,    VM::AVX2, false>(V); }
//TEST_P(BitonicMachineAVX2_i64,    BitonicSortAVX2Desc) { perform_bitonic_machine_sort_test<int64_t,  VM::AVX2, false>(V); }
//TEST_P(BitonicMachineAVX2_ui64,   BitonicSortAVX2Desc) { perform_bitonic_machine_sort_test<uint64_t, VM::AVX2, false>(V); }
//TEST_P(BitonicMachineAVX2_double, BitonicSortAVX2Desc) { perform_bitonic_machine_sort_test<double,   VM::AVX2, false>(V); }

struct BitonicAVX2_i16    : public SortTest<int16_t> {};
struct BitonicAVX2_ui16   : public SortTest<uint16_t> {};
struct BitonicAVX2_i32    : public SortTest<int32_t> {};
struct BitonicAVX2_ui32   : public SortTest<uint32_t> {};
struct BitonicAVX2_float  : public SortTest<float> {};
struct BitonicAVX2_i64    : public SortTest<int64_t> {};
struct BitonicAVX2_ui64   : public SortTest<uint64_t> {};
struct BitonicAVX2_double : public SortTest<double> {};

auto bitonic_allvalues_avx2_16 = ValuesIn(range(1, 8192, 1));
auto bitonic_allvalues_avx2_32 = ValuesIn(range(1, 4096, 1));
auto bitonic_allvalues_avx2_64 = ValuesIn(range(1, 2048, 1));

INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_i16,    bitonic_allvalues_avx2_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_ui16,   bitonic_allvalues_avx2_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_i32,    bitonic_allvalues_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_ui32,   bitonic_allvalues_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_float,  bitonic_allvalues_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_i64,    bitonic_allvalues_avx2_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_ui64,   bitonic_allvalues_avx2_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_double, bitonic_allvalues_avx2_64, PrintValue());

TEST_P(BitonicAVX2_i32,    BitonicSortAVX2) { perform_bitonic_sort_test<int32_t,  VM::AVX2>(V); }
TEST_P(BitonicAVX2_ui32,   BitonicSortAVX2) { perform_bitonic_sort_test<uint32_t, VM::AVX2>(V); }
TEST_P(BitonicAVX2_float,  BitonicSortAVX2) { perform_bitonic_sort_test<float,    VM::AVX2>(V); }
TEST_P(BitonicAVX2_i64,    BitonicSortAVX2) { perform_bitonic_sort_test<int64_t,  VM::AVX2>(V); }
TEST_P(BitonicAVX2_ui64,   BitonicSortAVX2) { perform_bitonic_sort_test<uint64_t, VM::AVX2>(V); }
TEST_P(BitonicAVX2_double, BitonicSortAVX2) { perform_bitonic_sort_test<double,   VM::AVX2>(V); }

}
#include "vxsort_targets_disable.h"
