#include "vxsort_targets_enable_avx512.h"

#include "BM_packunpack.h"

#include <packer.h>
#include <vector_machine/machine_traits.avx512.h>


namespace vxsort_bench {
using namespace vxsort::types;
using vxsort::vector_machine;

BENCHMARK_TEMPLATE(BM_pack, int64_t, int32_t, vector_machine::AVX512, 3, 1)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_pack, int64_t, int32_t, vector_machine::AVX512, 3, 2)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_pack, int64_t, int32_t, vector_machine::AVX512, 3, 4)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_unpack, int64_t, int32_t, vector_machine::AVX512, 3, 1)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_unpack, int64_t, int32_t, vector_machine::AVX512, 3, 2)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_unpack, int64_t, int32_t, vector_machine::AVX512, 3, 4)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(benchmark::kMillisecond)->ThreadRange(1, processor_count);

}

#include "vxsort_targets_disable.h"
