#include "vxsort_targets_enable_avx2.h"

#include <benchmark/benchmark.h>
#include <vxsort.avx2.h>
#include <random>

#include "BM_fullsort.vxsort.h"

namespace vxsort_bench {
using namespace vxsort::types;
using vm = vxsort::vector_machine;

void register_fullsort_avx2_i_benchmarks() {
    register_fullsort_benchmarks<vm::AVX2, 8, i16, i32, i64>();
}

}  // namespace vxsort_bench

#include "vxsort_targets_disable.h"
