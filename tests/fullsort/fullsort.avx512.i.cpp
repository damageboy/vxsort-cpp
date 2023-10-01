#include "vxsort_targets_enable_avx512.h"

#include "gtest/gtest.h"

#include <vxsort.avx512.h>
#include "fullsort_test.h"
#include "../sort_fixtures.h"

namespace vxsort_tests {
using namespace vxsort::types;
using testing::Types;

using VM = vxsort::vector_machine;
using namespace vxsort;

void register_fullsort_avx512_i_tests() {
    register_fullsort_benchmarks<VM::AVX512, 8, i16>(10, 10000,   10, 0x1000, 0x1);
    register_fullsort_benchmarks<VM::AVX512, 8, i32>(10, 1000000, 10, 0x1000, 0x1);
    register_fullsort_benchmarks<VM::AVX512, 8, i64>(10, 1000000, 10, 0x1000, 0x1);
}

}

#include "vxsort_targets_disable.h"
