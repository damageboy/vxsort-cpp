#include "vxsort_targets_enable_avx2.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <smallsort/avx2/bitonic_machine.avx2.h>
#include <vector_machine/machine_traits.avx2.h>

#include "smallsort_test.h"
#include "../sort_fixtures.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

struct BitonicMachineAVX2_i16    : public SortFixture<i16> {};
struct BitonicMachineAVX2_i32    : public SortFixture<i32> {};
struct BitonicMachineAVX2_i64    : public SortFixture<i64> {};
struct BitonicMachineAVX2_ui16   : public SortFixture<u16> {};
struct BitonicMachineAVX2_ui32   : public SortFixture<u32> {};
struct BitonicMachineAVX2_ui64   : public SortFixture<u64> {};
struct BitonicMachineAVX2_float  : public SortFixture<f32> {};
struct BitonicMachineAVX2_double : public SortFixture<f64> {};

auto bitonic_machine_allvalues_avx2_16 = ValuesIn(range(16, 64, 16));
auto bitonic_machine_allvalues_avx2_32 = ValuesIn(range(8, 32, 8));
auto bitonic_machine_allvalues_avx2_64 = ValuesIn(range(4, 16, 4));

INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_i16,    bitonic_machine_allvalues_avx2_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_i32,    bitonic_machine_allvalues_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_i64,    bitonic_machine_allvalues_avx2_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_ui16,   bitonic_machine_allvalues_avx2_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_ui32,   bitonic_machine_allvalues_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_ui64,   bitonic_machine_allvalues_avx2_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_float,  bitonic_machine_allvalues_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_double, bitonic_machine_allvalues_avx2_64, PrintValue());

TEST_P(BitonicMachineAVX2_i16,    BitonicSortAVX2Asc) { bitonic_machine_sort_test<i16, VM::AVX2, true>(V); }
TEST_P(BitonicMachineAVX2_i32,    BitonicSortAVX2Asc) { bitonic_machine_sort_test<i32, VM::AVX2, true>(V); }
TEST_P(BitonicMachineAVX2_i64,    BitonicSortAVX2Asc) { bitonic_machine_sort_test<i64, VM::AVX2, true>(V); }
TEST_P(BitonicMachineAVX2_ui16,   BitonicSortAVX2Asc) { bitonic_machine_sort_test<u16, VM::AVX2, true>(V); }
TEST_P(BitonicMachineAVX2_ui32,   BitonicSortAVX2Asc) { bitonic_machine_sort_test<u32, VM::AVX2, true>(V); }
TEST_P(BitonicMachineAVX2_ui64,   BitonicSortAVX2Asc) { bitonic_machine_sort_test<u64, VM::AVX2, true>(V); }
TEST_P(BitonicMachineAVX2_float,  BitonicSortAVX2Asc) { bitonic_machine_sort_test<f32, VM::AVX2, true>(V); }
TEST_P(BitonicMachineAVX2_double, BitonicSortAVX2Asc) { bitonic_machine_sort_test<f64, VM::AVX2, true>(V); }

//TEST_P(BitonicMachineAVX2_i32,    BitonicSortAVX2Desc) { bitonic_machine_sort_test<i32, VM::AVX2, false>(V); }
//TEST_P(BitonicMachineAVX2_ui32,   BitonicSortAVX2Desc) { bitonic_machine_sort_test<u32, VM::AVX2, false>(V); }
//TEST_P(BitonicMachineAVX2_i64,    BitonicSortAVX2Desc) { bitonic_machine_sort_test<i64, VM::AVX2, false>(V); }
//TEST_P(BitonicMachineAVX2_ui64,   BitonicSortAVX2Desc) { bitonic_machine_sort_test<u64, VM::AVX2, false>(V); }
//TEST_P(BitonicMachineAVX2_float,  BitonicSortAVX2Desc) { bitonic_machine_sort_test<f32, VM::AVX2, false>(V); }
//TEST_P(BitonicMachineAVX2_double, BitonicSortAVX2Desc) { bitonic_machine_sort_test<f64, VM::AVX2, false>(V); }

struct BitonicAVX2_i16 : public SortFixture<i16> {};
struct BitonicAVX2_i32 : public SortFixture<i32> {};
struct BitonicAVX2_i64 : public SortFixture<i64> {};
struct BitonicAVX2_u16 : public SortFixture<u16> {};
struct BitonicAVX2_u32 : public SortFixture<u32> {};
struct BitonicAVX2_u64 : public SortFixture<u64> {};
struct BitonicAVX2_f32 : public SortFixture<f32> {};
struct BitonicAVX2_f64 : public SortFixture<f64> {};

auto bitonic_allvalues_avx2_16 = ValuesIn(range(1, 8192, 1));
auto bitonic_allvalues_avx2_32 = ValuesIn(range(1, 4096, 1));
auto bitonic_allvalues_avx2_64 = ValuesIn(range(1, 2048, 1));

INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_i16, bitonic_allvalues_avx2_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_i32, bitonic_allvalues_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_i64, bitonic_allvalues_avx2_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_u16, bitonic_allvalues_avx2_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_u32, bitonic_allvalues_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_u64, bitonic_allvalues_avx2_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_f32, bitonic_allvalues_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_f64, bitonic_allvalues_avx2_64, PrintValue());

TEST_P(BitonicAVX2_i16, BitonicSortAVX2) { bitonic_sort_test<i16, VM::AVX2>(V); }
TEST_P(BitonicAVX2_i32, BitonicSortAVX2) { bitonic_sort_test<i32, VM::AVX2>(V); }
TEST_P(BitonicAVX2_i64, BitonicSortAVX2) { bitonic_sort_test<i64, VM::AVX2>(V); }
TEST_P(BitonicAVX2_u16, BitonicSortAVX2) { bitonic_sort_test<u16, VM::AVX2>(V); }
TEST_P(BitonicAVX2_u32, BitonicSortAVX2) { bitonic_sort_test<u32, VM::AVX2>(V); }
TEST_P(BitonicAVX2_u64, BitonicSortAVX2) { bitonic_sort_test<u64, VM::AVX2>(V); }
TEST_P(BitonicAVX2_f32, BitonicSortAVX2) { bitonic_sort_test<f32, VM::AVX2>(V); }
TEST_P(BitonicAVX2_f64, BitonicSortAVX2) { bitonic_sort_test<f64, VM::AVX2>(V); }

}
#include "vxsort_targets_disable.h"
