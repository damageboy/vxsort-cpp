#include "vxsort_targets_enable_avx512.h"

#include <pack_machine.h>
#include <vector_machine/machine_traits.avx512.h>

#include "pack_machine_test.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

template<typename T>
using PackMachineAVX512Test = PackMachineTest<T, VM::AVX512>;

using PackTypes = ::testing::Types<i16, u16, i32, u32, i64, u64>;
TYPED_TEST_SUITE(PackMachineAVX512Test, PackTypes);

TYPED_TEST(PackMachineAVX512Test, PackingWorks) {
    test_packunpack<TypeParam , VM::AVX512>(this);
}

};

#include "vxsort_targets_disable.h"
