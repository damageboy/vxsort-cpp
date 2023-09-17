#include "vxsort_targets_enable_avx2.h"

#include "gtest/gtest.h"

#include <smallsort/bitonic_sort.avx2.h>

#include "smallsort_test.h"
#include "../sort_fixtures.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

#ifdef VXSORT_TEST_AVX2_I16
auto bitonic_machine_allvalues_avx2_i16 = ValuesIn(SortTestParams<i16>::gen_step(SortPattern::unique_values, 16, 64, 16, 0, 0x1000, 0x1));
auto bitonic_allvalues_avx2_i16 = ValuesIn(SortTestParams<i16>::gen_step(SortPattern::unique_values, 1, 8192, 1, 0, 0x1000, 0x1));
struct BitonicMachineAVX2_i16    : public ParametrizedSortFixture<i16> {};
struct BitonicAVX2_i16 : public ParametrizedSortFixture<i16> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_i16, bitonic_machine_allvalues_avx2_i16, PrintSortTestParams<i16>());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_i16, bitonic_allvalues_avx2_i16, PrintSortTestParams<i16>());
#endif

#ifdef VXSORT_TEST_AVX2_I32
auto bitonic_machine_allvalues_avx2_i32 = ValuesIn(SortTestParams<i32>::gen_step(SortPattern::unique_values, 8, 32, 8, 0, 0x1000, 0x1));
auto bitonic_allvalues_avx2_i32 = ValuesIn(SortTestParams<i32>::gen_step(SortPattern::unique_values, 1, 4096, 1, 0, 0x1000, 0x1));
struct BitonicMachineAVX2_i32: public ParametrizedSortFixture<i32> {};
struct BitonicAVX2_i32 : public ParametrizedSortFixture<i32> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_i32, bitonic_machine_allvalues_avx2_i32, PrintSortTestParams<i32>());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_i32, bitonic_allvalues_avx2_i32, PrintSortTestParams<i32>());
#endif

#ifdef VXSORT_TEST_AVX2_I64
auto bitonic_machine_allvalues_avx2_i64 = ValuesIn(SortTestParams<i64>::gen_step(SortPattern::unique_values, 4, 16, 4, 0, 0x1000, 0x1));
auto bitonic_allvalues_avx2_i64 = ValuesIn(SortTestParams<i64>::gen_step(SortPattern::unique_values, 1, 2048, 1, 0, 0x1000, 0x1));
struct BitonicMachineAVX2_i64    : public ParametrizedSortFixture<i64> {};
struct BitonicAVX2_i64 : public ParametrizedSortFixture<i64> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_i64, bitonic_machine_allvalues_avx2_i64, PrintSortTestParams<i64>());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_i64, bitonic_allvalues_avx2_i64, PrintSortTestParams<i64>());
#endif

#ifdef VXSORT_TEST_AVX2_U16
auto bitonic_machine_allvalues_avx2_u16 = ValuesIn(SortTestParams<u16>::gen_step(SortPattern::unique_values, 16, 64, 16, 0, 0x1000, 0x1));
auto bitonic_allvalues_avx2_u16 = ValuesIn(SortTestParams<u16>::gen_step(SortPattern::unique_values, 1, 8192, 1, 0, 0x1000, 0x1));
struct BitonicMachineAVX2_u16   : public ParametrizedSortFixture<u16> {};
struct BitonicAVX2_u16 : public ParametrizedSortFixture<u16> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_u16, bitonic_machine_allvalues_avx2_u16, PrintSortTestParams<u16>());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_u16, bitonic_allvalues_avx2_u16, PrintSortTestParams<u16>());
#endif

#ifdef VXSORT_TEST_AVX2_U32
auto bitonic_machine_allvalues_avx2_u32 = ValuesIn(SortTestParams<u32>::gen_step(SortPattern::unique_values, 8, 32, 8, 0, 0x1000, 0x1));
auto bitonic_allvalues_avx2_u32 = ValuesIn(SortTestParams<u32>::gen_step(SortPattern::unique_values, 1, 4096, 1, 0, 0x1000, 0x1));
struct BitonicMachineAVX2_u32   : public ParametrizedSortFixture<u32> {};
struct BitonicAVX2_u32 : public ParametrizedSortFixture<u32> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_u32, bitonic_machine_allvalues_avx2_u32, PrintSortTestParams<u32>());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_u32, bitonic_allvalues_avx2_u32, PrintSortTestParams<u32>());
#endif

#ifdef VXSORT_TEST_AVX2_U64
auto bitonic_machine_allvalues_avx2_u64 = ValuesIn(SortTestParams<u64>::gen_step(SortPattern::unique_values, 4, 16, 4, 0, 0x1000, 0x1));
auto bitonic_allvalues_avx2_u64 = ValuesIn(SortTestParams<u64>::gen_step(SortPattern::unique_values, 1, 2048, 1, 0, 0x1000, 0x1));
struct BitonicMachineAVX2_u64   : public ParametrizedSortFixture<u64> {};
struct BitonicAVX2_u64 : public ParametrizedSortFixture<u64> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_u64, bitonic_machine_allvalues_avx2_u64, PrintSortTestParams<u64>());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_u64, bitonic_allvalues_avx2_u64, PrintSortTestParams<u64>());
#endif

#ifdef VXSORT_TEST_AVX2_F32
auto bitonic_machine_allvalues_avx2_f32 = ValuesIn(SortTestParams<f32>::gen_step(SortPattern::unique_values, 8, 32, 8, 0, 1234.5f, 0.1f));
auto bitonic_allvalues_avx2_f32 = ValuesIn(SortTestParams<f32>::gen_step(SortPattern::unique_values, 1, 4096, 1, 0, 1234.5f, 0.1f));
struct BitonicMachineAVX2_f32 : public ParametrizedSortFixture<f32> {};
struct BitonicAVX2_f32 : public ParametrizedSortFixture<f32> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_f32, bitonic_machine_allvalues_avx2_f32, PrintSortTestParams<f32>());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_f32, bitonic_allvalues_avx2_f32, PrintSortTestParams<f32>());
#endif

#ifdef VXSORT_TEST_AVX2_F64
auto bitonic_machine_allvalues_avx2_f64 = ValuesIn(SortTestParams<f64>::gen_step(SortPattern::unique_values, 4, 16, 4, 0, 1234.5, 0.1));
auto bitonic_allvalues_avx2_f64 = ValuesIn(SortTestParams<f64>::gen_step(SortPattern::unique_values, 1, 2048, 1, 0, 1234.5, 0.1));
struct BitonicMachineAVX2_f64 : public ParametrizedSortFixture<f64> {};
struct BitonicAVX2_f64 : public ParametrizedSortFixture<f64> {};
INSTANTIATE_TEST_SUITE_P(BitonicMachineAVX2, BitonicMachineAVX2_f64, bitonic_machine_allvalues_avx2_f64, PrintSortTestParams<f64>());
INSTANTIATE_TEST_SUITE_P(BitonicAVX2, BitonicAVX2_f64, bitonic_allvalues_avx2_f64, PrintSortTestParams<f64>());
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
