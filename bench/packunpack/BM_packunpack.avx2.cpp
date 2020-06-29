#include "vxsort_targets_enable_avx2.h"

#include "BM_packunpack.h"

#include <packer.h>
#include <vector_machine/machine_traits.avx2.h>

using vxsort::vector_machine;

namespace vxsort_bench {
using namespace vxsort::types;
using benchmark::TimeUnit;
using vm = vxsort::vector_machine;


BENCHMARK_TEMPLATE(BM_pack, i64, i32, vm::AVX2, 3, 1)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_pack, i64, i32, vm::AVX2, 3, 2)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_pack, i64, i32, vm::AVX2, 3, 4)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(kMillisecond)->ThreadRange(1, processor_count);

BENCHMARK_TEMPLATE(BM_unpack, i64, i32, vm::AVX2, 3, 1)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_unpack, i64, i32, vm::AVX2, 3, 2)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(kMillisecond)->ThreadRange(1, processor_count);
BENCHMARK_TEMPLATE(BM_unpack, i64, i32, vm::AVX2, 3, 4)->RangeMultiplier(2)->Range(MIN_PACKUNPACK, MAX_PACKUNPACK)->Unit(kMillisecond)->ThreadRange(1, processor_count);

}

#include "vxsort_targets_disable.h"
