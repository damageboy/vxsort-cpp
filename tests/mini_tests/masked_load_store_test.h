#ifndef VXSORT_MASKED_LOAD_STORE_TEST_H
#define VXSORT_MASKED_LOAD_STORE_TEST_H

#include <gtest/gtest.h>
#include "mini_fixtures.h"

#include "defs.h"
#include "vector_machine/machine_traits.h"
#include "isa_detection.h"
#include "alignment.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

template <typename T, VM M>
void test_prefix_mask_load_on_page_boundary(PageWithLavaBoundariesFixture<T, M> *fixture)
{
    if (!::vxsort::supports_vector_machine(M)) {
        GTEST_SKIP_("Current CPU does not support the minimal features for this test");
        return;
    }

    using VMT = vxsort::vxsort_machine_traits<T, M>;
    static constexpr auto MAX = std::numeric_limits<T>::max();
    const auto MAXV = VMT::broadcast(MAX);

    for (auto w = 1; w < VMT::N; w++) {
        auto mask = VMT::generate_prefix_mask(w);
        auto *load_addr = fixture->page_with_data + fixture->num_elements - w;
        auto result = VMT::load_partial_vec((typename VMT::TV *) load_addr, MAXV, mask);
        auto &res_array = reinterpret_cast<T(&)[VMT::N]>(result);

        for (auto i = 0; i < w; ++i)
            ASSERT_EQ(res_array[i], fixture->get_expected_value(load_addr + i));
        for (auto i = w; i < VMT::N; ++i)
            ASSERT_EQ(res_array[i], MAX);
    }
}

template <typename T, VM M>
void test_suffix_mask_load_on_page_boundary(PageWithLavaBoundariesFixture<T, M> *fixture)
{
    if (!::vxsort::supports_vector_machine(M)) {
        GTEST_SKIP_("Current CPU does not support the minimal features for this test");
        return;
    }

    using VMT = vxsort::vxsort_machine_traits<T, M>;
    static constexpr auto MAX = std::numeric_limits<T>::max();
    const auto MAXV = VMT::broadcast(MAX);

    for (auto w = 1; w < VMT::N; w++) {
        auto mask = VMT::generate_suffix_mask(w);
        auto *load_addr = fixture->page_with_data - w;
        auto result = VMT::load_partial_vec((typename VMT::TV *) load_addr, MAXV, mask);
        auto &res_array = reinterpret_cast<T(&)[VMT::N]>(result);

        for (auto i = 0; i < w; ++i)
            ASSERT_EQ(res_array[i], MAX);
        for (auto i = w; i < VMT::N; ++i)
            ASSERT_EQ(res_array[i], fixture->get_expected_value(load_addr + i));
    }
}

template <typename T, VM M>
void test_left_alignment_and_masked_loads(PageWithLavaBoundariesFixture<T, M> *fixture)
{
    if (!::vxsort::supports_vector_machine(M)) {
        GTEST_SKIP_("Current CPU does not support the minimal features for this test");
        return;
    }

    using VMT = vxsort::vxsort_machine_traits<T, M>;
    using AH = vxsort::alignment_hint<T, M>;

    static constexpr auto MAX = std::numeric_limits<T>::max();
    const auto MAXV = VMT::broadcast(MAX);

    for (auto w = 0; w < VMT::N; w++) {
        auto *load_addr = fixture->page_with_data + w;

        AH align;
        align.calc_left_alignment(load_addr);
        auto mask = VMT::generate_suffix_mask(align.left_masked_amount);

        load_addr -= align.left_masked_amount;

        ASSERT_TRUE(AH::is_aligned(load_addr));

        auto result = VMT::load_partial_vec((typename VMT::TV *) load_addr, MAXV, mask);
        auto &res_array = reinterpret_cast<T(&)[VMT::N]>(result);

        for (auto i = 0; i < align.left_masked_amount; ++i)
            ASSERT_EQ(res_array[i], MAX);
        for (auto i = align.left_masked_amount; i < VMT::N; ++i)
            ASSERT_EQ(res_array[i], fixture->get_expected_value(load_addr + i));
    }
}

template <typename T, VM M>
void test_right_alignment_and_masked_loads(PageWithLavaBoundariesFixture<T, M> *fixture)
{
    if (!::vxsort::supports_vector_machine(M)) {
        GTEST_SKIP_("Current CPU does not support the minimal features for this test");
        return;
    }

    using VMT = vxsort::vxsort_machine_traits<T, M>;
    using AH = vxsort::alignment_hint<T, M>;

    static constexpr auto MAX = std::numeric_limits<T>::max();
    const auto MAXV = VMT::broadcast(MAX);

    for (auto w = 0; w < VMT::N; w++) {
        auto *load_addr = fixture->page_with_data + fixture->num_elements - w;

        AH align;
        align.calc_right_alignment(load_addr);

        load_addr -= align.right_unmasked_amount;

        ASSERT_TRUE(AH::is_aligned(load_addr));

        auto mask = VMT::generate_prefix_mask(align.right_unmasked_amount);
        auto result = VMT::load_partial_vec((typename VMT::TV *) load_addr, MAXV, mask);
        auto &res_array = reinterpret_cast<T(&)[VMT::N]>(result);

        for (auto i = 0; i < align.right_unmasked_amount; ++i)
            ASSERT_EQ(res_array[i], fixture->get_expected_value(load_addr + i));
        for (auto i = align.right_unmasked_amount; i < VMT::N; ++i)
            ASSERT_EQ(res_array[i], MAX);
    }
}


};

#endif //VXSORT_MASKED_LOAD_STORE_TEST_H
