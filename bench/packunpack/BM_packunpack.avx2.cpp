#include "vxsort_targets_enable_avx2.h"

#include "BM_packunpack.h"

#include <machine_traits.avx2.h>
#include <packer.h>

using vxsort::vector_machine;

namespace vxsort_bench {

BENCHMARK_TEMPLATE(BM_pack, int64_t, int32_t, vector_machine::AVX2, 3, 1)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_pack, int64_t, int32_t, vector_machine::AVX2, 3, 2)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_pack, int64_t, int32_t, vector_machine::AVX2, 3, 4)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_unpack, int64_t, int32_t, vector_machine::AVX2, 3, 1)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_unpack, int64_t, int32_t, vector_machine::AVX2, 3, 2)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_unpack, int64_t, int32_t, vector_machine::AVX2, 3, 4)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

}

#include "vxsort_targets_disable.h"
