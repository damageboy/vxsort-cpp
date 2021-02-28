#include "vxsort_targets_enable_avx512.h"

#include <random>
#include <benchmark/benchmark.h>

#include <smallsort/avx512/bitonic_machine.AVX512.double.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.float.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.int32_t.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.int64_t.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.uint32_t.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.uint64_t.generated.h>
#include <vector_machine/machine_traits.avx512.h>

#include "BM_fullsort.h"

using vxsort::vector_machine;

namespace vxsort_bench {

BENCHMARK_TEMPLATE(BM_vxsort_strided, int64_t, vector_machine::AVX512,  1)->RangeMultiplier(2)->Range(MIN_STRIDE, MAX_STRIDE)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort_strided, int64_t, vector_machine::AVX512,  4)->RangeMultiplier(2)->Range(MIN_STRIDE, MAX_STRIDE)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort_strided, int64_t, vector_machine::AVX512,  8)->RangeMultiplier(2)->Range(MIN_STRIDE, MAX_STRIDE)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort_strided, int64_t, vector_machine::AVX512, 12)->RangeMultiplier(2)->Range(MIN_STRIDE, MAX_STRIDE)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

}

#include "vxsort_targets_disable.h"
