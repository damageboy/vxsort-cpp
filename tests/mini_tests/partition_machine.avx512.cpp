#include "vxsort_targets_enable_avx512.h"

#include <gtest/gtest.h>

#include <partition_machine.avx512.h>
#include "partition_machine_test.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

template<typename T>
using PartitionMachineAVX512Test = PageWithLavaBoundariesFixture<T, VM::AVX512>;

using TestTypes = ::testing::Types<
#ifdef VXSORT_TEST_AVX512_I
            i16, i32, i64
#endif
#ifdef VXSORT_TEST_AVX512_U
            u16, u32, u64
#endif
#ifdef VXSORT_TEST_AVX512_F
            f32, f64
#endif
>;
TYPED_TEST_SUITE(PartitionMachineAVX512Test, TestTypes);

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
