#include "vxsort_targets_enable_avx512.h"

#include <partition_machine.avx512.h>
#include <vector_machine/machine_traits.avx512.h>

#include "partition_machine_test.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

template<typename T>
using PartitionMachineAVX512Test = PageWithLavaBoundariesFixture<T, VM::AVX512>;

using MaskedTypes = ::testing::Types<i16, u16, i32, u32, i64, u64>;
TYPED_TEST_SUITE(PartitionMachineAVX512Test, MaskedTypes);

TYPED_TEST(PartitionMachineAVX512Test, PartitioningWorks) {
    test_partition<TypeParam , VM::AVX512>(this);
}

TYPED_TEST(PartitionMachineAVX512Test, PartitioningIsStable) {
    test_partition_stability<TypeParam , VM::AVX512>(this);
}


TYPED_TEST(PartitionMachineAVX512Test, PartitionAlignmentWorks) {
    test_partition_alignment<TypeParam , VM::AVX512>(this);
}

};

#include "vxsort_targets_disable.h"
