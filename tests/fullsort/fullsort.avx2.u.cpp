#include "vxsort_targets_enable_avx2.h"

#include "gtest/gtest.h"

#include <vxsort.avx2.h>
#include "fullsort_test.h"

namespace vxsort_tests {
using namespace vxsort::types;
using testing::Types;

using VM = vxsort::vector_machine;
using namespace vxsort;

void register_fullsort_avx2_u_tests() {
    register_fullsort_benchmarks<VM::AVX2, 8, u16>(10, 10000,   10, 0x1000, 0x1);
    register_fullsort_benchmarks<VM::AVX2, 8, u32>(10, 1000000, 10, 0x1000, 0x1);
    register_fullsort_benchmarks<VM::AVX2, 8, u64>(10, 1000000, 10, 0x1000, 0x1);
}

}


#include "vxsort_targets_disable.h"
