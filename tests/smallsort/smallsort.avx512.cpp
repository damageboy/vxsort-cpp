#include "vxsort_targets_enable_avx512.h"

#include "gtest/gtest.h"

#include <smallsort/bitonic_sort.avx512.h>

#include "smallsort_test.h"
#include "../sort_fixtures.h"

namespace vxsort_tests {
using namespace vxsort::types;
using testing::Types;

using VM = vxsort::vector_machine;

auto bitonic_machine_allvalues_avx512_16 = ValuesIn(range(32, 128, 32));
auto bitonic_machine_allvalues_avx512_32 = ValuesIn(range(16, 64, 16));
auto bitonic_machine_allvalues_avx512_64 = ValuesIn(range(8, 32, 8));
auto bitonic_allvalues_avx512_16 = ValuesIn(range(1, 8192, 1));
auto bitonic_allvalues_avx512_32 = ValuesIn(range(1, 4096, 1));
auto bitonic_allvalues_avx512_64 = ValuesIn(range(1, 2048, 1));

#ifdef VXSORT_TEST_AVX512_I16
struct BitonicMachineAVX512_i16 : public SortFixture<i16> {};
struct BitonicAVX512_i16 : public SortFixture<i16> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_i16, bitonic_machine_allvalues_avx512_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_i16, bitonic_allvalues_avx512_16, PrintValue());
#endif

#ifdef VXSORT_TEST_AVX512_I32
struct BitonicMachineAVX512_i32 : public SortFixture<i32> {};
struct BitonicAVX512_i32 : public SortFixture<i32> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_i32, bitonic_machine_allvalues_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_i32, bitonic_allvalues_avx512_32, PrintValue());
#endif

#ifdef VXSORT_TEST_AVX512_I64
struct BitonicMachineAVX512_i64 : public SortFixture<i64> {};
struct BitonicAVX512_i64 : public SortFixture<i64> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_i64, bitonic_machine_allvalues_avx512_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_i64, bitonic_allvalues_avx512_64, PrintValue());
#endif

#ifdef VXSORT_TEST_AVX512_U16
struct BitonicMachineAVX512_u16 : public SortFixture<u16> {};
struct BitonicAVX512_u16 : public SortFixture<u16> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_u16, bitonic_machine_allvalues_avx512_16, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_u16, bitonic_allvalues_avx512_16, PrintValue());
#endif

#ifdef VXSORT_TEST_AVX512_U32
struct BitonicMachineAVX512_u32 : public SortFixture<u32> {};
struct BitonicAVX512_u32 : public SortFixture<u32> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_u32, bitonic_machine_allvalues_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_u32, bitonic_allvalues_avx512_32, PrintValue());
#endif

#ifdef VXSORT_TEST_AVX512_U64
struct BitonicMachineAVX512_u64 : public SortFixture<u64> {};
struct BitonicAVX512_u64 : public SortFixture<u64> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_u64, bitonic_machine_allvalues_avx512_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_u64, bitonic_allvalues_avx512_64, PrintValue());
#endif

#ifdef VXSORT_TEST_AVX512_F32
struct BitonicMachineAVX512_f32 : public SortFixture<f32> {};
struct BitonicAVX512_f32 : public SortFixture<f32> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_f32, bitonic_machine_allvalues_avx512_32, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_f32, bitonic_allvalues_avx512_32, PrintValue());
#endif

#ifdef VXSORT_TEST_AVX512_F64
struct BitonicMachineAVX512_f64 : public SortFixture<f64> {};
struct BitonicAVX512_f64 : public SortFixture<f64> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX512, BitonicMachineAVX512_f64, bitonic_machine_allvalues_avx512_64, PrintValue());
INSTANTIATE_TEST_SUITE_P(BitonicAVX512, BitonicAVX512_f64, bitonic_allvalues_avx512_64, PrintValue());
#endif


#ifdef VXSORT_TEST_AVX512_I16
TEST_P(BitonicMachineAVX512_i16, BitonicSortAVX512Asc) { bitonic_machine_sort_test<i16, VM::AVX512, true>(V); }
TEST_P(BitonicAVX512_i16, BitonicSortAVX512) { bitonic_sort_test<i16, VM::AVX512>(V); }
#endif

#ifdef VXSORT_TEST_AVX512_I32
TEST_P(BitonicMachineAVX512_i32, BitonicSortAVX512Asc) { bitonic_machine_sort_test<i32, VM::AVX512, true>(V); }
TEST_P(BitonicAVX512_i32, BitonicSortAVX512) { bitonic_sort_test<i32, VM::AVX512>(V); }
#endif

#ifdef VXSORT_TEST_AVX512_I64
TEST_P(BitonicMachineAVX512_i64, BitonicSortAVX512Asc) { bitonic_machine_sort_test<i64, VM::AVX512, true>(V); }
TEST_P(BitonicAVX512_i64, BitonicSortAVX512) { bitonic_sort_test<i64, VM::AVX512>(V); }
#endif

#ifdef VXSORT_TEST_AVX512_U16
TEST_P(BitonicMachineAVX512_u16, BitonicSortAVX512Asc) { bitonic_machine_sort_test<u16, VM::AVX512, true>(V); }
TEST_P(BitonicAVX512_u16, BitonicSortAVX512) { bitonic_sort_test<u16, VM::AVX512>(V); }
#endif

#ifdef VXSORT_TEST_AVX512_U32
TEST_P(BitonicMachineAVX512_u32, BitonicSortAVX512Asc) { bitonic_machine_sort_test<u32, VM::AVX512, true>(V); }
TEST_P(BitonicAVX512_u32, BitonicSortAVX512) { bitonic_sort_test<u32, VM::AVX512>(V); }
#endif

#ifdef VXSORT_TEST_AVX512_U64
TEST_P(BitonicMachineAVX512_u64, BitonicSortAVX512Asc) { bitonic_machine_sort_test<u64, VM::AVX512, true>(V); }
TEST_P(BitonicAVX512_u64, BitonicSortAVX512) { bitonic_sort_test<u64, VM::AVX512>(V); }
#endif

#ifdef VXSORT_TEST_AVX512_F32
TEST_P(BitonicMachineAVX512_f32, BitonicSortAVX512Asc) { bitonic_machine_sort_test<f32, VM::AVX512, true>(V); }
TEST_P(BitonicAVX512_f32, BitonicSortAVX512) { bitonic_sort_test<f32, VM::AVX512>(V); }
#endif

#ifdef VXSORT_TEST_AVX512_F64
TEST_P(BitonicMachineAVX512_f64, BitonicSortAVX512Asc) { bitonic_machine_sort_test<f64, VM::AVX512, true>(V); }
TEST_P(BitonicAVX512_f64, BitonicSortAVX512) { bitonic_sort_test<f64, VM::AVX512>(V); }
#endif

//TEST_P(BitonicMachineAVX512_i32, BitonicSortAVX2Desc) { bitonic_machine_sort_test<i32, VM::AVX512, false>(V); }
//TEST_P(BitonicMachineAVX512_u32, BitonicSortAVX2Desc) { bitonic_machine_sort_test<u32, VM::AVX512, false>(V); }
//TEST_P(BitonicMachineAVX512_f32, BitonicSortAVX2Desc) { bitonic_machine_sort_test<f32, VM::AVX512, false>(V); }
//TEST_P(BitonicMachineAVX512_i64, BitonicSortAVX2Desc) { bitonic_machine_sort_test<i64, VM::AVX512, false>(V); }
//TEST_P(BitonicMachineAVX512_u64, BitonicSortAVX2Desc) { bitonic_machine_sort_test<u64, VM::AVX512, false>(V); }
//TEST_P(BitonicMachineAVX512_f64, BitonicSortAVX2Desc) { bitonic_machine_sort_test<f64, VM::AVX512, false>(V); }
}

#include "vxsort_targets_disable.h"
