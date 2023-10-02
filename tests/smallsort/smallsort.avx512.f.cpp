#include "vxsort_targets_enable_avx512.h"

#include "gtest/gtest.h"

#include <smallsort/bitonic_sort.avx512.h>
#include "smallsort_test.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

void register_smallsort_avx512_f_tests() {
    register_bitonic_tests<VM::AVX512, f32>(16*1024, 1234.5, 0.1);
    register_bitonic_tests<VM::AVX512, f64>(16*1024, 1234.5, 0.1);

    register_bitonic_machine_tests<VM::AVX512, f32>(1234.5, 0.1);
    register_bitonic_machine_tests<VM::AVX512, f64>(1234.5, 0.1);
}
}

#include "vxsort_targets_disable.h"
