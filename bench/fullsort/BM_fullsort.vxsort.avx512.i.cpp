#include "vxsort_targets_enable_avx512.h"

#include <benchmark/benchmark.h>
#include <random>

#include <vxsort.avx512.h>

#include "BM_fullsort.vxsort.h"

namespace vxsort_bench {
using namespace vxsort::types;
using vm = vxsort::vector_machine;

void register_fullsort_avx512_i_benchmarks() {
    register_fullsort_benchmarks<vm::AVX512, 8, i16, i32, i64>();
}

}  // namespace vxsort_bench

#include "vxsort_targets_disable.h"
