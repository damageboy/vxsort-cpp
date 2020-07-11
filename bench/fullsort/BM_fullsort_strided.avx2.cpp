#include "vxsort_targets_enable_avx2.h"

#include <random>
#include <benchmark/benchmark.h>

#include <machine_traits.avx2.h>
#include <smallsort/bitonic_sort.AVX2.int32_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.uint32_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.float.generated.h>
#include <smallsort/bitonic_sort.AVX2.int64_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.uint64_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.double.generated.h>

#include "BM_fullsort.h"

using vxsort::vector_machine;

namespace vxsort_bench {

BENCHMARK_TEMPLATE(BM_vxsort_strided, int64_t, vector_machine::AVX2,  1)->RangeMultiplier(2)->Range(MIN_STRIDE, MAX_STRIDE)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort_strided, int64_t, vector_machine::AVX2,  4)->RangeMultiplier(2)->Range(MIN_STRIDE, MAX_STRIDE)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort_strided, int64_t, vector_machine::AVX2,  8)->RangeMultiplier(2)->Range(MIN_STRIDE, MAX_STRIDE)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort_strided, int64_t, vector_machine::AVX2, 12)->RangeMultiplier(2)->Range(MIN_STRIDE, MAX_STRIDE)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

}

#include "vxsort_targets_disable.h"
