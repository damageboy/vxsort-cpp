#include "vxsort_targets_enable_avx512.h"

#include <random>
#include <benchmark/benchmark.h>

#include <smallsort/avx512/bitonic_machine.AVX512.i16.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.i32.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.i64.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.u16.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.u32.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.u64.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.f32.generated.h>
#include <smallsort/avx512/bitonic_machine.AVX512.f64.generated.h>
#include <vector_machine/machine_traits.avx512.h>

#include "BM_fullsort.h"

namespace vxsort_bench {
using namespace vxsort::types;

BENCHMARK_TEMPLATE(BM_vxsort, i16, vector_machine::AVX512,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i16, vector_machine::AVX512,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i16, vector_machine::AVX512,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i16, vector_machine::AVX512, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_vxsort, i32, vector_machine::AVX512,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i32, vector_machine::AVX512,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i32, vector_machine::AVX512,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i32, vector_machine::AVX512, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_vxsort, u32, vector_machine::AVX512,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, u32, vector_machine::AVX512,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, u32, vector_machine::AVX512,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, u32, vector_machine::AVX512, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_vxsort, f32, vector_machine::AVX512,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, f32, vector_machine::AVX512,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, f32, vector_machine::AVX512,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, f32, vector_machine::AVX512, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_vxsort, i64, vector_machine::AVX512,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i64, vector_machine::AVX512,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i64, vector_machine::AVX512,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, i64, vector_machine::AVX512, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_vxsort, u64, vector_machine::AVX512,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, u64, vector_machine::AVX512,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, u64, vector_machine::AVX512,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, u64, vector_machine::AVX512, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_vxsort, f64, vector_machine::AVX512,  1)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, f64, vector_machine::AVX512,  4)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, f64, vector_machine::AVX512,  8)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_vxsort, f64, vector_machine::AVX512, 12)->RangeMultiplier(2)->Range(MIN_SORT, MAX_SORT)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

}

#include "vxsort_targets_disable.h"
