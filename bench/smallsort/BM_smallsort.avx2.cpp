#include "vxsort_targets_enable_avx2.h"

#include "BM_smallsort.h"

#include <smallsort/avx2/bitonic_machine.AVX2.double.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.float.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.int16_t.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.int32_t.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.int64_t.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.uint16_t.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.uint32_t.generated.h>
#include <smallsort/avx2/bitonic_machine.AVX2.uint64_t.generated.h>
#include <vector_machine/machine_traits.avx2.h>

using namespace vxsort;

namespace vxsort_bench {

BENCHMARK(BM_insertionsort)->DenseRange(4, 64, 4)->Unit(benchmark::kNanosecond);

BENCHMARK_TEMPLATE(BM_bitonic_sort, i16, vector_machine::AVX2)->DenseRange(8, 4096, 8)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, u16, vector_machine::AVX2)->DenseRange(8, 4096, 8)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, i32, vector_machine::AVX2)->DenseRange(4, 2048, 4)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, u32, vector_machine::AVX2)->DenseRange(4, 2048, 4)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, f32, vector_machine::AVX2)->DenseRange(4, 2048, 4)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, i64, vector_machine::AVX2)->DenseRange(2, 1024, 2)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, u64, vector_machine::AVX2)->DenseRange(2, 1024, 2)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_bitonic_sort, f64, vector_machine::AVX2)->DenseRange(2, 1024, 2)->Unit(benchmark::kNanosecond);

}

#include "vxsort_targets_disable.h"
