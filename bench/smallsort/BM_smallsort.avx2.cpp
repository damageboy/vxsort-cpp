#include "vxsort_targets_enable_avx2.h"

#include "BM_smallsort.h"

#include <smallsort/avx2/bitonic_machine.avx2.h>
#include <vector_machine/machine_traits.avx2.h>


namespace vxsort_bench {
using namespace vxsort::types;
using benchmark::TimeUnit;
using vm = vxsort::vector_machine;

BENCHMARK_TEMPLATE(BM_bitonic_sort, i16, vm::AVX2)->DenseRange(16, 4096, 8)->Unit(kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, u16, vm::AVX2)->DenseRange(16, 4096, 8)->Unit(kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, i32, vm::AVX2)->DenseRange( 4, 2048, 4)->Unit(kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, u32, vm::AVX2)->DenseRange( 4, 2048, 4)->Unit(kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, f32, vm::AVX2)->DenseRange( 4, 2048, 4)->Unit(kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, i64, vm::AVX2)->DenseRange( 2, 1024, 2)->Unit(kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, u64, vm::AVX2)->DenseRange( 2, 1024, 2)->Unit(kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, f64, vm::AVX2)->DenseRange( 2, 1024, 2)->Unit(kNanosecond);

}

#include "vxsort_targets_disable.h"
