#include "vxsort_targets_enable_avx2.h"

#include "gtest/gtest.h"

#include <vxsort.avx2.h>
#include "fullsort_test.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

void register_fullsort_avx2_f_tests() {
    register_fullsort_tests<VM::AVX2, 8, f32>(10, 1000000, 10, 1234.5, 0.1);
    register_fullsort_tests<VM::AVX2, 8, f32>(10, 1000000, 10, 1234.5, 0.1);
}
}

#include "vxsort_targets_disable.h"
