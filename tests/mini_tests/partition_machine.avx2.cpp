#include "vxsort_targets_enable_avx2.h"

#include <gtest/gtest.h>

#include <partition_machine.avx2.h>
#include "partition_machine_test.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

template<typename T>
using PartitionMachineAVX2Test = PageWithLavaBoundariesFixture<T, VM::AVX2>;

using TestTypes = ::testing::Types<
#ifdef VXSORT_TEST_AVX2_I
            i16, i32, i64
#endif
#ifdef VXSORT_TEST_AVX2_U
            u16, u32, u64
#endif
#ifdef VXSORT_TEST_AVX2_F
            f32, f64
#endif
    >;
TYPED_TEST_SUITE(PartitionMachineAVX2Test, TestTypes);

TYPED_TEST(PartitionMachineAVX2Test, PartitioningWorks) {
    test_partition<TypeParam , VM::AVX2>(this);
}

TYPED_TEST(PartitionMachineAVX2Test, PartitioningIsStable) {
    test_partition_stability<TypeParam , VM::AVX2>(this);
}


TYPED_TEST(PartitionMachineAVX2Test, PartitionAlignmentWorks) {
    test_partition_alignment<TypeParam , VM::AVX2>(this);
}

};

#include "vxsort_targets_disable.h"
