#ifndef VXSORT_PARTITION_MACHINE_TEST_H
#define VXSORT_PARTITION_MACHINE_TEST_H

#include <gtest/gtest.h>

#include <fmt/format.h>
#include "mini_fixtures.h"

#include "defs.h"
#include "vector_machine/machine_traits.h"
#include "isa_detection.h"
#include "alignment.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

template <typename T, VM M>
void test_partition(PageWithLavaBoundariesFixture<T, M> *fixture)
{
    if (!::vxsort::supports_vector_machine(M)) {
        GTEST_SKIP_("Current CPU does not support the minimal features for this test");
        return;
    }

    using VMT = vxsort::vxsort_machine_traits<T, M>;
    using PM = vxsort::partition_machine<T, M>;

    for (auto p = 0; p < VMT::N; p++) {
        auto *load_addr = fixture->page_with_data;
        auto pivot = fixture->get_expected_value(load_addr + p) - 1;

        auto PV = VMT::broadcast(pivot);

        auto data = VMT::load_vec((typename VMT::TV *) load_addr);

        T spill_left[VMT::N*2];
        T spill_right[VMT::N*2];

        T* RESTRICT spill_left_ptr = spill_left;
        T* RESTRICT spill_right_ptr = spill_right + VMT::N;
        T* RESTRICT spill_right_start_ptr = spill_right_ptr;

        memset(spill_left, 0x66, sizeof(spill_left));
        memset(spill_right, 0x66, sizeof(spill_right));

        PM::partition_block(data, PV, spill_left_ptr, spill_right_ptr);

        ASSERT_EQ(spill_left_ptr - spill_left, p);
        ASSERT_EQ(spill_right_start_ptr - spill_right_ptr, VMT::N - p);

        for (auto i = 0; i < p; ++i) {
            auto expected_value = fixture->get_expected_value(load_addr + i);
            ASSERT_EQ(spill_left[i], expected_value);
        }

        for (auto i = VMT::N - 1; i >= p; --i) {
            auto expected_value = fixture->get_expected_value(load_addr + i);
            ASSERT_EQ(spill_right_start_ptr[i], expected_value);

        }

    }
}

};

#endif //VXSORT_PARTITION_MACHINE_TEST_H
