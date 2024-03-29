#include "vxsort_targets_enable_avx512.h"

#include <random>
#include <benchmark/benchmark.h>

#include <vxsort.avx512.h>

#include "BM_fullsort.vxsort.h"

namespace vxsort_bench {
using namespace vxsort::types;
using benchmark::TimeUnit;
using vm = vxsort::vector_machine;

BENCHMARK_TEMPLATE(BM_vxsort, i16, vm::AVX512,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i16, vm::AVX512,  2)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i16, vm::AVX512,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i16, vm::AVX512,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_vxsort, i32, vm::AVX512,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i32, vm::AVX512,  2)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i32, vm::AVX512,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i32, vm::AVX512,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_vxsort, i64, vm::AVX512,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i64, vm::AVX512,  2)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i64, vm::AVX512,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i64, vm::AVX512,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);

}

#include "vxsort_targets_disable.h"
