#include "vxsort_targets_enable_avx2.h"

#include <random>
#include <benchmark/benchmark.h>

#include <vxsort.avx2.h>

#include "BM_fullsort.h"

namespace vxsort_bench {
using namespace vxsort::types;
using benchmark::TimeUnit;
using vm = vxsort::vector_machine;

BENCHMARK_TEMPLATE(BM_vxsort, i32, vm::AVX2,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i32, vm::AVX2,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i32, vm::AVX2,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i32, vm::AVX2, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_vxsort, u32, vm::AVX2,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, u32, vm::AVX2,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, u32, vm::AVX2,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, u32, vm::AVX2, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_vxsort, i64, vm::AVX2,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i64, vm::AVX2,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i64, vm::AVX2,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i64, vm::AVX2, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_vxsort, u64, vm::AVX2,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, u64, vm::AVX2,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, u64, vm::AVX2,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, u64, vm::AVX2, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_vxsort, f32, vm::AVX2,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, f32, vm::AVX2,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, f32, vm::AVX2,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, f32, vm::AVX2, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_vxsort, f64, vm::AVX2,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, f64, vm::AVX2,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, f64, vm::AVX2,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, f64, vm::AVX2, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(kMillisecond)->ThreadRange(1, processor_count);

}

#include "vxsort_targets_disable.h"
