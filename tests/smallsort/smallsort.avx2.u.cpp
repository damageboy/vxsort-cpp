#include "vxsort_targets_enable_avx2.h"

#include "gtest/gtest.h"

#include <smallsort/bitonic_sort.avx2.h>
#include "smallsort_test.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

void register_smallsort_avx2_u_tests() {
    register_bitonic_tests<VM::AVX2, u16>(16*1024, 0x1000, 0x1);
    register_bitonic_tests<VM::AVX2, u32>(16*1024, 0x1000, 0x1);
    register_bitonic_tests<VM::AVX2, u64>(16*1024, 0x1000, 0x1);

    register_bitonic_machine_tests<VM::AVX2, u16>(0x1000, 0x1);
    register_bitonic_machine_tests<VM::AVX2, u32>(0x1000, 0x1);
    register_bitonic_machine_tests<VM::AVX2, u64>(0x1000, 0x1);
}
}

#include "vxsort_targets_disable.h"
