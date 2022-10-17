#include "vxsort_targets_enable_avx2.h"

#include <partition_machine.avx2.h>

#include "partition_machine_test.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

template<typename T>
using PartitionMachineAVX2Test = PageWithLavaBoundariesFixture<T, VM::AVX2>;

using MaskedTypes = ::testing::Types</*i16, u16, */i32, u32, i64, u64>;
TYPED_TEST_SUITE(PartitionMachineAVX2Test, MaskedTypes);

TYPED_TEST(PartitionMachineAVX2Test, PartitioningWorks) {
    test_partition<TypeParam , VM::AVX2>(this);
}

TYPED_TEST(PartitionMachineAVX2Test, PartitionAlignmentWorks) {
    test_partition_alignment<TypeParam , VM::AVX2>(this);
}

};

#include "vxsort_targets_disable.h"