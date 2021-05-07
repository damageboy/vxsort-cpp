#include "vxsort_targets_enable_avx2.h"

#include "BM_smallsort.h"

#include <smallsort/avx2/bitonic_machine.AVX2.h>
#include <vector_machine/machine_traits.avx2.h>


namespace vxsort_bench {
using namespace vxsort::types;

BENCHMARK(BM_insertionsort)->DenseRange(4, 64, 4)->Unit(benchmark::kNanosecond);

BENCHMARK_TEMPLATE(BM_bitonic_sort, i16, vector_machine::AVX2)->DenseRange(16, 4096, 8)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, u16, vector_machine::AVX2)->DenseRange(16, 4096, 8)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, i32, vector_machine::AVX2)->DenseRange(4, 2048, 4)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, u32, vector_machine::AVX2)->DenseRange(4, 2048, 4)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, f32, vector_machine::AVX2)->DenseRange(4, 2048, 4)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, i64, vector_machine::AVX2)->DenseRange(2, 1024, 2)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, u64, vector_machine::AVX2)->DenseRange(2, 1024, 2)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, f64, vector_machine::AVX2)->DenseRange(2, 1024, 2)->Unit(benchmark::kNanosecond);

}

#include "vxsort_targets_disable.h"
