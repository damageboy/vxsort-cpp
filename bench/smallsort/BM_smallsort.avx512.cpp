#include "vxsort_targets_enable_avx512.h"

#include "BM_smallsort.h"

#include <smallsort/avx512/bitonic_machine.AVX512.f64.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.f32.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.i16.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.i32.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.i64.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.u16.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.u32.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.u64.generated.h>
#include <vector_machine/machine_traits.avx512.h>

namespace vxsort_bench {
using namespace vxsort::types;

BENCHMARK(BM_insertionsort)->DenseRange(4, 64, 4)->Unit(benchmark::kNanosecond);

BENCHMARK_TEMPLATE(BM_bitonic_sort, i16, vector_machine::AVX512)->DenseRange(8, 4096, 8)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, u16, vector_machine::AVX512)->DenseRange(8, 4096, 8)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, i32, vector_machine::AVX512)->DenseRange(4, 2048, 4)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, u32, vector_machine::AVX512)->DenseRange(4, 2048, 4)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, f32, vector_machine::AVX512)->DenseRange(4, 2048, 4)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, i64, vector_machine::AVX512)->DenseRange(2, 1024, 2)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, u64, vector_machine::AVX512)->DenseRange(2, 1024, 2)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, f64, vector_machine::AVX512)->DenseRange(2, 1024, 2)->Unit(benchmark::kNanosecond);

}

#include "vxsort_targets_disable.h"
