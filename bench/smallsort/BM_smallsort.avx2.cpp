#include "BM_smallsort.h"

#include <smallsort/bitonic_sort.AVX2.double.generated.h>
#include <smallsort/bitonic_sort.AVX2.float.generated.h>
#include <smallsort/bitonic_sort.AVX2.int32_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.int64_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.uint32_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.uint64_t.generated.h>

using vxsort::vector_machine;

namespace vxsort_bench {

BENCHMARK(BM_insertionsort)->DenseRange(4, 64, 4)->Unit(benchmark::kNanosecond);

BENCHMARK_TEMPLATE(BM_bitonic_sort, int32_t,  vector_machine::AVX2)->DenseRange(8, 128, 8)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, uint32_t, vector_machine::AVX2)->DenseRange(2, 128, 4)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, float,    vector_machine::AVX2)->DenseRange(2, 128, 4)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, int64_t,  vector_machine::AVX2)->DenseRange(2,  64, 2)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, uint64_t, vector_machine::AVX2)->DenseRange(2,  64, 2)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, double,   vector_machine::AVX2)->DenseRange(2,  64, 2)->Unit(benchmark::kNanosecond);

}

