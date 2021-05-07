#include "vxsort_targets_enable_avx2.h"

#include <random>
#include <benchmark/benchmark.h>

#include <smallsort/avx2/bitonic_machine.AVX2.h>
#include <vector_machine/machine_traits.avx2.h>

#include "BM_fullsort.h"

using namespace vxsort;

namespace vxsort_bench {

BENCHMARK_TEMPLATE(BM_vxsort_strided, i64, vector_machine::AVX2,  1)->RangeMultiplier(2)->Range(MIN_STRIDE, MAX_STRIDE)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort_strided, i64, vector_machine::AVX2,  4)->RangeMultiplier(2)->Range(MIN_STRIDE, MAX_STRIDE)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort_strided, i64, vector_machine::AVX2,  8)->RangeMultiplier(2)->Range(MIN_STRIDE, MAX_STRIDE)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort_strided, i64, vector_machine::AVX2, 12)->RangeMultiplier(2)->Range(MIN_STRIDE, MAX_STRIDE)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

}

#include "vxsort_targets_disable.h"
