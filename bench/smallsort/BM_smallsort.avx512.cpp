#include "vxsort_targets_enable_avx512.h"

#include "BM_smallsort.h"

#include <smallsort/bitonic_sort.avx512.h>

namespace vxsort_bench {
using namespace vxsort::types;
using benchmark::TimeUnit;
using vm = vxsort::vector_machine;

BENCHMARK_TEMPLATE(BM_bitonic_sort, i16, vm::AVX512)->DenseRange(8, 4096, 8)->Unit(kNanosecond)->MinTime(0.1);
BENCHMARK_TEMPLATE(BM_bitonic_sort, u16, vm::AVX512)->DenseRange(8, 4096, 8)->Unit(kNanosecond)->MinTime(0.1);
BENCHMARK_TEMPLATE(BM_bitonic_sort, i32, vm::AVX512)->DenseRange(4, 2048, 4)->Unit(kNanosecond)->MinTime(0.1);
BENCHMARK_TEMPLATE(BM_bitonic_sort, u32, vm::AVX512)->DenseRange(4, 2048, 4)->Unit(kNanosecond)->MinTime(0.1);
BENCHMARK_TEMPLATE(BM_bitonic_sort, f32, vm::AVX512)->DenseRange(4, 2048, 4)->Unit(kNanosecond)->MinTime(0.1);
BENCHMARK_TEMPLATE(BM_bitonic_sort, i64, vm::AVX512)->DenseRange(2, 1024, 2)->Unit(kNanosecond)->MinTime(0.1);
BENCHMARK_TEMPLATE(BM_bitonic_sort, u64, vm::AVX512)->DenseRange(2, 1024, 2)->Unit(kNanosecond)->MinTime(0.1);
BENCHMARK_TEMPLATE(BM_bitonic_sort, f64, vm::AVX512)->DenseRange(2, 1024, 2)->Unit(kNanosecond)->MinTime(0.1);

BENCHMARK_TEMPLATE(BM_bitonic_machine, i16, vm::AVX512, 2)->Unit(kNanosecond)->MinTime(0.1);
BENCHMARK_TEMPLATE(BM_bitonic_machine, u16, vm::AVX512, 2)->Unit(kNanosecond)->MinTime(0.1);
BENCHMARK_TEMPLATE(BM_bitonic_machine, i32, vm::AVX512, 2)->Unit(kNanosecond)->MinTime(0.1);
BENCHMARK_TEMPLATE(BM_bitonic_machine, u32, vm::AVX512, 2)->Unit(kNanosecond)->MinTime(0.1);
BENCHMARK_TEMPLATE(BM_bitonic_machine, i64, vm::AVX512, 2)->Unit(kNanosecond)->MinTime(0.1);
BENCHMARK_TEMPLATE(BM_bitonic_machine, u64, vm::AVX512, 2)->Unit(kNanosecond)->MinTime(0.1);
BENCHMARK_TEMPLATE(BM_bitonic_machine, f32, vm::AVX512, 2)->Unit(kNanosecond)->MinTime(0.1);
BENCHMARK_TEMPLATE(BM_bitonic_machine, f64, vm::AVX512, 2)->Unit(kNanosecond)->MinTime(0.1);

}

#include "vxsort_targets_disable.h"
