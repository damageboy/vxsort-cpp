#include "vxsort_targets_enable_avx2.h"

#include <random>
#include <benchmark/benchmark.h>

#include <smallsort/avx2/bitonic_machine.AVX2.i32.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.i64.generated.h>
#include <vector_machine/machine_traits.avx2.h>

#include "BM_packnsort.h"

using vxsort::vector_machine;

namespace vxsort_bench {

BENCHMARK_TEMPLATE(BM_packnvxsort, vector_machine::AVX2, 3,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_packnvxsort, vector_machine::AVX2, 3,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_packnvxsort, vector_machine::AVX2, 3,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_packnvxsort, vector_machine::AVX2, 3, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
}

#include "vxsort_targets_disable.h"
