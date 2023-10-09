#include "vxsort_targets_enable_avx512.h"

#include "gtest/gtest.h"

#include <vxsort.avx512.h>
#include "fullsort_test.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

void register_fullsort_avx512_u_tests() {
    register_fullsort_tests<VM::AVX512, 8, u16>(10, 10000, 10, 0x1000, 0x1);
    register_fullsort_tests<VM::AVX512, 8, u32>(10, 1000000, 10, 0x1000, 0x1);
    register_fullsort_tests<VM::AVX512, 8, u64>(10, 1000000, 10, 0x1000, 0x1);
}
}

#include "vxsort_targets_disable.h"