#include "vxsort_targets_enable_avx2.h"

#include <gtest/gtest.h>

#include <vector_machine/machine_traits.avx2.h>
#include <pack_machine.h>

#include "pack_machine_test.h"


namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

template<typename T>
using PackMachineAVX2Test = PackMachineTest<T, VM::AVX2>;

using TestTypes = ::testing::Types<
#ifdef VXSORT_TEST_AVX2_I16
            i16, i32, i64
#endif
#ifdef VXSORT_TEST_AVX2_U16
            u16, u32, u64
#endif
#ifdef VXSORT_TEST_AVX2_F32
            f32, f64
#endif
    >;
TYPED_TEST_SUITE(PackMachineAVX2Test, TestTypes);

TYPED_TEST(PackMachineAVX2Test, PackingWorks) {
    test_packunpack<TypeParam , VM::AVX2>(this);
}

};

#include "vxsort_targets_disable.h"
