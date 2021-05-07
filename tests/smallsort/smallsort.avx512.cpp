#include "vxsort_targets_enable_avx512.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "../fixtures.h"

#include <smallsort/avx512/bitonic_machine.AVX512.h>
#include <smallsort/bitonic_machine.h>
#include <vector_machine/machine_traits.avx512.h>

#include "smallsort_test.h"

namespace vxsort_tests {
using testing::Types;

using VM = vxsort::vector_machine;

struct BitonicMachineAVX512_i16    : public SortTest<int16_t> {};
struct BitonicMachineAVX512_i32    : public SortTest<int32_t> {};
struct BitonicMachineAVX512_i64    : public SortTest<int64_t> {};
struct BitonicMachineAVX512_ui16   : public SortTest<uint16_t> {};
struct BitonicMachineAVX512_ui32   : public SortTest<uint32_t> {};
struct BitonicMachineAVX512_ui64   : public SortTest<uint64_t> {};
struct BitonicMachineAVX512_float  : public SortTest<float> {};
struct BitonicMachineAVX512_double : public SortTest<double> {};

auto bitonic_machine_allvalues_avx512_16 = ValuesIn(range(32, 128, 32));
auto bitonic_machine_allvalues_avx512_32 = ValuesIn(range(16, 64, 16));
auto bitonic_machine_allvalues_avx512_64 = ValuesIn(range(8, 32, 8));

INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_i16,    bitonic_machine_allvalues_avx512_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_i32,    bitonic_machine_allvalues_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_i64,    bitonic_machine_allvalues_avx512_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_ui16,   bitonic_machine_allvalues_avx512_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_ui32,   bitonic_machine_allvalues_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_ui64,   bitonic_machine_allvalues_avx512_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_float,  bitonic_machine_allvalues_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_double, bitonic_machine_allvalues_avx512_64, PrintValue());

TEST_P(BitonicMachineAVX512_i16,    BitonicSortAVX512Asc) { perform_bitonic_machine_sort_test<int16_t,  VM::AVX512, true>(V); }
TEST_P(BitonicMachineAVX512_i32,    BitonicSortAVX512Asc) { perform_bitonic_machine_sort_test<int32_t,  VM::AVX512, true>(V); }
TEST_P(BitonicMachineAVX512_i64,    BitonicSortAVX512Asc) { perform_bitonic_machine_sort_test<int64_t,  VM::AVX512, true>(V); }
TEST_P(BitonicMachineAVX512_ui16,   BitonicSortAVX512Asc) { perform_bitonic_machine_sort_test<uint16_t, VM::AVX512, true>(V); }
TEST_P(BitonicMachineAVX512_ui32,   BitonicSortAVX512Asc) { perform_bitonic_machine_sort_test<uint32_t, VM::AVX512, true>(V); }
TEST_P(BitonicMachineAVX512_ui64,   BitonicSortAVX512Asc) { perform_bitonic_machine_sort_test<uint64_t, VM::AVX512, true>(V); }
TEST_P(BitonicMachineAVX512_float,  BitonicSortAVX512Asc) { perform_bitonic_machine_sort_test<float,    VM::AVX512, true>(V); }
TEST_P(BitonicMachineAVX512_double, BitonicSortAVX512Asc) { perform_bitonic_machine_sort_test<double,   VM::AVX512, true>(V); }

//TEST_P(BitonicMachineAVX512_i32,    BitonicSortAVX2Desc) { perform_bitonic_machine_sort_test<int32_t,  VM::AVX512, false>(V); }
//TEST_P(BitonicMachineAVX512_ui32,   BitonicSortAVX2Desc) { perform_bitonic_machine_sort_test<uint32_t, VM::AVX512, false>(V); }
//TEST_P(BitonicMachineAVX512_float,  BitonicSortAVX2Desc) { perform_bitonic_machine_sort_test<float,    VM::AVX512, false>(V); }
//TEST_P(BitonicMachineAVX512_i64,    BitonicSortAVX2Desc) { perform_bitonic_machine_sort_test<int64_t,  VM::AVX512, false>(V); }
//TEST_P(BitonicMachineAVX512_ui64,   BitonicSortAVX2Desc) { perform_bitonic_machine_sort_test<uint64_t, VM::AVX512, false>(V); }
//TEST_P(BitonicMachineAVX512_double, BitonicSortAVX2Desc) { perform_bitonic_machine_sort_test<double,   VM::AVX512, false>(V); }

struct BitonicAVX512_i16    : public SortTest<int16_t> {};
struct BitonicAVX512_i32    : public SortTest<int32_t> {};
struct BitonicAVX512_i64    : public SortTest<int64_t> {};
struct BitonicAVX512_ui16   : public SortTest<uint16_t> {};
struct BitonicAVX512_ui32   : public SortTest<uint32_t> {};
struct BitonicAVX512_ui64   : public SortTest<uint64_t> {};
struct BitonicAVX512_float  : public SortTest<float> {};
struct BitonicAVX512_double : public SortTest<double> {};

auto bitonic_allvalues_avx512_16 = ValuesIn(range(1, 8192, 1));
auto bitonic_allvalues_avx512_32 = ValuesIn(range(1, 4096, 1));
auto bitonic_allvalues_avx512_64 = ValuesIn(range(1, 2048, 1));

INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_i16,    bitonic_allvalues_avx512_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_i32,    bitonic_allvalues_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_i64,    bitonic_allvalues_avx512_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_ui16,   bitonic_allvalues_avx512_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_ui32,   bitonic_allvalues_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_ui64,   bitonic_allvalues_avx512_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_float,  bitonic_allvalues_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_double, bitonic_allvalues_avx512_64, PrintValue());

TEST_P(BitonicAVX512_i16,    BitonicSortAVX512) { perform_bitonic_sort_test<int16_t,  VM::AVX512>(V); }
TEST_P(BitonicAVX512_i32,    BitonicSortAVX512) { perform_bitonic_sort_test<int32_t,  VM::AVX512>(V); }
TEST_P(BitonicAVX512_i64,    BitonicSortAVX512) { perform_bitonic_sort_test<int64_t,  VM::AVX512>(V); }
TEST_P(BitonicAVX512_ui16,   BitonicSortAVX512) { perform_bitonic_sort_test<uint16_t, VM::AVX512>(V); }
TEST_P(BitonicAVX512_ui32,   BitonicSortAVX512) { perform_bitonic_sort_test<uint32_t, VM::AVX512>(V); }
TEST_P(BitonicAVX512_ui64,   BitonicSortAVX512) { perform_bitonic_sort_test<uint64_t, VM::AVX512>(V); }
TEST_P(BitonicAVX512_float,  BitonicSortAVX512) { perform_bitonic_sort_test<float,    VM::AVX512>(V); }
TEST_P(BitonicAVX512_double, BitonicSortAVX512) { perform_bitonic_sort_test<double,   VM::AVX512>(V); }

}

#include "vxsort_targets_disable.h"
