#include "vxsort_targets_enable_avx2.h"

#include "gtest/gtest.h"

#include <smallsort/bitonic_sort.avx2.h>

#include "smallsort_test.h"
#include "../sort_fixtures.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

auto bitonic_machine_allvalues_avx2_16 = ValuesIn(range(16, 64, 16));
auto bitonic_machine_allvalues_avx2_32 = ValuesIn(range(8, 32, 8));
auto bitonic_machine_allvalues_avx2_64 = ValuesIn(range(4, 16, 4));

auto bitonic_allvalues_avx2_16 = ValuesIn(range(1, 8192, 1));
auto bitonic_allvalues_avx2_32 = ValuesIn(range(1, 4096, 1));
auto bitonic_allvalues_avx2_64 = ValuesIn(range(1, 2048, 1));

#ifdef VXSORT_TEST_AVX2_I16
struct BitonicMachineAVX2_i16    : public SortFixture<i16> {};
struct BitonicAVX2_i16 : public SortFixture<i16> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_i16,    bitonic_machine_allvalues_avx2_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_i16, bitonic_allvalues_avx2_16, PrintValue());
#endif

#ifdef VXSORT_TEST_AVX2_I32
struct BitonicMachineAVX2_i32    : public SortFixture<i32> {};
struct BitonicAVX2_i32 : public SortFixture<i32> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_i32,    bitonic_machine_allvalues_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_i32, bitonic_allvalues_avx2_32, PrintValue());
#endif

#ifdef VXSORT_TEST_AVX2_I64
struct BitonicMachineAVX2_i64    : public SortFixture<i64> {};
struct BitonicAVX2_i64 : public SortFixture<i64> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_i64,    bitonic_machine_allvalues_avx2_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_i64, bitonic_allvalues_avx2_64, PrintValue());
#endif

#ifdef VXSORT_TEST_AVX2_U16
struct BitonicMachineAVX2_u16   : public SortFixture<u16> {};
struct BitonicAVX2_u16 : public SortFixture<u16> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_u16,   bitonic_machine_allvalues_avx2_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_u16, bitonic_allvalues_avx2_16, PrintValue());
#endif

#ifdef VXSORT_TEST_AVX2_U32
struct BitonicMachineAVX2_u32   : public SortFixture<u32> {};
struct BitonicAVX2_u32 : public SortFixture<u32> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_u32,   bitonic_machine_allvalues_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_u32, bitonic_allvalues_avx2_32, PrintValue());
#endif

#ifdef VXSORT_TEST_AVX2_U64
struct BitonicMachineAVX2_u64   : public SortFixture<u64> {};
struct BitonicAVX2_u64 : public SortFixture<u64> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_u64,   bitonic_machine_allvalues_avx2_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_u64, bitonic_allvalues_avx2_64, PrintValue());
#endif

#ifdef VXSORT_TEST_AVX2_F32
struct BitonicMachineAVX2_f32 : public SortFixture<f32> {};
struct BitonicAVX2_f32 : public SortFixture<f32> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_f32,  bitonic_machine_allvalues_avx2_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_f32, bitonic_allvalues_avx2_32, PrintValue());
#endif

#ifdef VXSORT_TEST_AVX2_F64
struct BitonicMachineAVX2_f64 : public SortFixture<f64> {};
struct BitonicAVX2_f64 : public SortFixture<f64> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_f64, bitonic_machine_allvalues_avx2_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_f64, bitonic_allvalues_avx2_64, PrintValue());
#endif

#ifdef VXSORT_TEST_AVX2_I16
TEST_P(BitonicMachineAVX2_i16,    BitonicSortAVX2Asc) { bitonic_machine_sort_test<i16, VM::AVX2, true>(V); }
TEST_P(BitonicAVX2_i16, BitonicSortAVX2) { bitonic_sort_test<i16, VM::AVX2>(V); }
#endif

#ifdef VXSORT_TEST_AVX2_I32
TEST_P(BitonicMachineAVX2_i32,    BitonicSortAVX2Asc) { bitonic_machine_sort_test<i32, VM::AVX2, true>(V); }
TEST_P(BitonicAVX2_i32, BitonicSortAVX2) { bitonic_sort_test<i32, VM::AVX2>(V); }
#endif

#ifdef VXSORT_TEST_AVX2_I64
TEST_P(BitonicMachineAVX2_i64,    BitonicSortAVX2Asc) { bitonic_machine_sort_test<i64, VM::AVX2, true>(V); }
TEST_P(BitonicAVX2_i64, BitonicSortAVX2) { bitonic_sort_test<i64, VM::AVX2>(V); }
#endif
#ifdef VXSORT_TEST_AVX2_U16
TEST_P(BitonicMachineAVX2_u16,   BitonicSortAVX2Asc) { bitonic_machine_sort_test<u16, VM::AVX2, true>(V); }
TEST_P(BitonicAVX2_u16, BitonicSortAVX2) { bitonic_sort_test<u16, VM::AVX2>(V); }
#endif

#ifdef VXSORT_TEST_AVX2_U32
TEST_P(BitonicMachineAVX2_u32,   BitonicSortAVX2Asc) { bitonic_machine_sort_test<u32, VM::AVX2, true>(V); }
TEST_P(BitonicAVX2_u32, BitonicSortAVX2) { bitonic_sort_test<u32, VM::AVX2>(V); }
#endif

#ifdef VXSORT_TEST_AVX2_U64
TEST_P(BitonicMachineAVX2_u64,   BitonicSortAVX2Asc) { bitonic_machine_sort_test<u64, VM::AVX2, true>(V); }
TEST_P(BitonicAVX2_u64, BitonicSortAVX2) { bitonic_sort_test<u64, VM::AVX2>(V); }
#endif

#ifdef VXSORT_TEST_AVX2_F32
TEST_P(BitonicMachineAVX2_f32,  BitonicSortAVX2Asc) { bitonic_machine_sort_test<f32, VM::AVX2, true>(V); }
TEST_P(BitonicAVX2_f32, BitonicSortAVX2) { bitonic_sort_test<f32, VM::AVX2>(V); }
#endif

#ifdef VXSORT_TEST_AVX2_F64
TEST_P(BitonicMachineAVX2_f64, BitonicSortAVX2Asc) { bitonic_machine_sort_test<f64, VM::AVX2, true>(V); }
TEST_P(BitonicAVX2_f64, BitonicSortAVX2) { bitonic_sort_test<f64, VM::AVX2>(V); }
#endif

//TEST_P(BitonicMachineAVX2_i32,    BitonicSortAVX2Desc) { bitonic_machine_sort_test<i32, VM::AVX2, false>(V); }
//TEST_P(BitonicMachineAVX2_u32,   BitonicSortAVX2Desc) { bitonic_machine_sort_test<u32, VM::AVX2, false>(V); }
//TEST_P(BitonicMachineAVX2_i64,    BitonicSortAVX2Desc) { bitonic_machine_sort_test<i64, VM::AVX2, false>(V); }
//TEST_P(BitonicMachineAVX2_u64,   BitonicSortAVX2Desc) { bitonic_machine_sort_test<u64, VM::AVX2, false>(V); }
//TEST_P(BitonicMachineAVX2_f32,  BitonicSortAVX2Desc) { bitonic_machine_sort_test<f32, VM::AVX2, false>(V); }
//TEST_P(BitonicMachineAVX2_f64, BitonicSortAVX2Desc) { bitonic_machine_sort_test<f64, VM::AVX2, false>(V); }

}
#include "vxsort_targets_disable.h"
