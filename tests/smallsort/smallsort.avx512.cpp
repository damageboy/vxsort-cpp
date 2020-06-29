#include "vxsort_targets_enable_avx512.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <smallsort/avx512/bitonic_machine.avx512.h>
#include <smallsort/bitonic_machine.h>
#include <vector_machine/machine_traits.avx512.h>

#include "smallsort_test.h"
#include "../fixtures.h"

namespace vxsort_tests {
using namespace vxsort::types;
using testing::Types;

using VMT = vxsort::vector_machine;

struct BitonicMachineAVX512_i16 : public SortTest<i16> {};
struct BitonicMachineAVX512_i32 : public SortTest<i32> {};
struct BitonicMachineAVX512_i64 : public SortTest<i64> {};
struct BitonicMachineAVX512_u16 : public SortTest<u16> {};
struct BitonicMachineAVX512_u32 : public SortTest<u32> {};
struct BitonicMachineAVX512_u64 : public SortTest<u64> {};
struct BitonicMachineAVX512_f32 : public SortTest<f32> {};
struct BitonicMachineAVX512_f64 : public SortTest<f64> {};

auto bitonic_machine_allvalues_avx512_16 = ValuesIn(range(32, 128, 32));
auto bitonic_machine_allvalues_avx512_32 = ValuesIn(range(16, 64, 16));
auto bitonic_machine_allvalues_avx512_64 = ValuesIn(range(8, 32, 8));

INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_i16, bitonic_machine_allvalues_avx512_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_i32, bitonic_machine_allvalues_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_i64, bitonic_machine_allvalues_avx512_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_u16, bitonic_machine_allvalues_avx512_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_u32, bitonic_machine_allvalues_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_u64, bitonic_machine_allvalues_avx512_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_f32, bitonic_machine_allvalues_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_f64, bitonic_machine_allvalues_avx512_64, PrintValue());

TEST_P(BitonicMachineAVX512_i16, BitonicSortAVX512Asc) { bitonic_machine_sort_test<i16, VMT::AVX512, true>(V); }
TEST_P(BitonicMachineAVX512_i32, BitonicSortAVX512Asc) { bitonic_machine_sort_test<i32, VMT::AVX512, true>(V); }
TEST_P(BitonicMachineAVX512_i64, BitonicSortAVX512Asc) { bitonic_machine_sort_test<i64, VMT::AVX512, true>(V); }
TEST_P(BitonicMachineAVX512_u16, BitonicSortAVX512Asc) { bitonic_machine_sort_test<u16, VMT::AVX512, true>(V); }
TEST_P(BitonicMachineAVX512_u32, BitonicSortAVX512Asc) { bitonic_machine_sort_test<u32, VMT::AVX512, true>(V); }
TEST_P(BitonicMachineAVX512_u64, BitonicSortAVX512Asc) { bitonic_machine_sort_test<u64, VMT::AVX512, true>(V); }
TEST_P(BitonicMachineAVX512_f32, BitonicSortAVX512Asc) { bitonic_machine_sort_test<f32, VMT::AVX512, true>(V); }
TEST_P(BitonicMachineAVX512_f64, BitonicSortAVX512Asc) { bitonic_machine_sort_test<f64, VMT::AVX512, true>(V); }

//TEST_P(BitonicMachineAVX512_i32, BitonicSortAVX2Desc) { bitonic_machine_sort_test<i32, VMT::AVX512, false>(V); }
//TEST_P(BitonicMachineAVX512_u32, BitonicSortAVX2Desc) { bitonic_machine_sort_test<u32, VMT::AVX512, false>(V); }
//TEST_P(BitonicMachineAVX512_f32, BitonicSortAVX2Desc) { bitonic_machine_sort_test<f32, VMT::AVX512, false>(V); }
//TEST_P(BitonicMachineAVX512_i64, BitonicSortAVX2Desc) { bitonic_machine_sort_test<i64, VMT::AVX512, false>(V); }
//TEST_P(BitonicMachineAVX512_u64, BitonicSortAVX2Desc) { bitonic_machine_sort_test<u64, VMT::AVX512, false>(V); }
//TEST_P(BitonicMachineAVX512_f64, BitonicSortAVX2Desc) { bitonic_machine_sort_test<f64, VMT::AVX512, false>(V); }

struct BitonicAVX512_i16 : public SortTest<i16> {};
struct BitonicAVX512_i32 : public SortTest<i32> {};
struct BitonicAVX512_i64 : public SortTest<i64> {};
struct BitonicAVX512_u16 : public SortTest<u16> {};
struct BitonicAVX512_u32 : public SortTest<u32> {};
struct BitonicAVX512_u64 : public SortTest<u64> {};
struct BitonicAVX512_f32 : public SortTest<f32> {};
struct BitonicAVX512_f64 : public SortTest<f64> {};

auto bitonic_allvalues_avx512_16 = ValuesIn(range(1, 8192, 1));
auto bitonic_allvalues_avx512_32 = ValuesIn(range(1, 4096, 1));
auto bitonic_allvalues_avx512_64 = ValuesIn(range(1, 2048, 1));

INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_i16, bitonic_allvalues_avx512_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_i32, bitonic_allvalues_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_i64, bitonic_allvalues_avx512_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_u16, bitonic_allvalues_avx512_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_u32, bitonic_allvalues_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_u64, bitonic_allvalues_avx512_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_f32, bitonic_allvalues_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_f64, bitonic_allvalues_avx512_64, PrintValue());

TEST_P(BitonicAVX512_i16, BitonicSortAVX512) { bitonic_sort_test<i16, VMT::AVX512>(V); }
TEST_P(BitonicAVX512_i32, BitonicSortAVX512) { bitonic_sort_test<i32, VMT::AVX512>(V); }
TEST_P(BitonicAVX512_i64, BitonicSortAVX512) { bitonic_sort_test<i64, VMT::AVX512>(V); }
TEST_P(BitonicAVX512_u16, BitonicSortAVX512) { bitonic_sort_test<u16, VMT::AVX512>(V); }
TEST_P(BitonicAVX512_u32, BitonicSortAVX512) { bitonic_sort_test<u32, VMT::AVX512>(V); }
TEST_P(BitonicAVX512_u64, BitonicSortAVX512) { bitonic_sort_test<u64, VMT::AVX512>(V); }
TEST_P(BitonicAVX512_f32, BitonicSortAVX512) { bitonic_sort_test<f32, VMT::AVX512>(V); }
TEST_P(BitonicAVX512_f64, BitonicSortAVX512) { bitonic_sort_test<f64, VMT::AVX512>(V); }

}

#include "vxsort_targets_disable.h"
