#include "vxsort_targets_enable_avx512.h"

#include <pack_machine.h>
#include <vector_machine/machine_traits.avx512.h>

#include "pack_machine_test.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

template<typename T>
using PackMachineAVX512Test = PackMachineTest<T, VM::AVX512>;

using TestTypes = ::testing::Types<
#ifdef VXSORT_TEST_AVX512_I16
            i16, i32, i64
#endif
#ifdef VXSORT_TEST_AVX512_U16
            u16, u32, u64
#endif
#ifdef VXSORT_TEST_AVX512_F32
            f32, f64
#endif
>;
TYPED_TEST_SUITE(PackMachineAVX512Test, TestTypes);

TYPED_TEST(PackMachineAVX512Test, PackingWorks) {
    test_packunpack<TypeParam , VM::AVX512>(this);
}

};

#include "vxsort_targets_disable.h"
