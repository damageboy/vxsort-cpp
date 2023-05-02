#ifndef VXSORT_PARTITION_MACHINE_TEST_H
#define VXSORT_PARTITION_MACHINE_TEST_H

#include <random>
#include <algorithm>
#include <span>
#include <fmt/format.h>
#include "mini_fixtures.h"

#include "defs.h"
#include "vector_machine/machine_traits.h"
#include "../test_isa.h"
#include "alignment.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

template <typename T, VM M>
void test_partition(PageWithLavaBoundariesFixture<T, M> *fixture)
{
    VXSORT_TEST_ISA();

    using VMT = vxsort::vxsort_machine_traits<T, M>;
    using PM = vxsort::partition_machine<T, M>;
    static constexpr auto N = VMT::N;

    for (auto p = 0; p < VMT::N; p++) {
        auto *load_addr = fixture->page_with_data;
        auto pivot = fixture->get_expected_value(load_addr + p) - 1;

        auto s = std::span<T>(load_addr, N);
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::shuffle(s.begin(), s.end(), gen);

        auto PV = VMT::broadcast(pivot);

        auto data = VMT::load_vec((typename VMT::TV *) load_addr);

        T spill_left[N*2];
        T spill_right[N*2];

        T* RESTRICT spill_left_end = spill_left;
        // partition_block expects the left/right *write* pointers to point
        // to the next vector write position, for right write pointer
        // this means N elements BEFORE the end of the spill buffer
        T* RESTRICT spill_right_start = spill_right + N;
        T* RESTRICT spill_right_end = spill_right_start;

        memset(spill_left, 0x66, sizeof(spill_left));
        memset(spill_right, 0x66, sizeof(spill_right));

        PM::partition_block(data, PV, spill_left_end, spill_right_end);

        ASSERT_EQ(spill_left_end - spill_left, p);
        ASSERT_EQ(spill_right_start - spill_right_end, N - p);

        for (auto i = 0; i < p; ++i) {
            ASSERT_TRUE(spill_left[i] <= pivot);
        }

        for (auto i = VMT::N - 1; i >= p; --i) {
            ASSERT_TRUE(spill_right_start[i] > pivot);

        }
    }
}


template <typename T, VM M>
void test_partition_stability(PageWithLavaBoundariesFixture<T, M> *fixture)
{
    VXSORT_TEST_ISA();

    using VMT = vxsort::vxsort_machine_traits<T, M>;
    using PM = vxsort::partition_machine<T, M>;
    static constexpr auto N = VMT::N;

    for (auto p = 0; p < VMT::N; p++) {
        auto *load_addr = fixture->page_with_data;
        auto pivot = fixture->get_expected_value(load_addr + p) - 1;

        auto PV = VMT::broadcast(pivot);

        auto data = VMT::load_vec((typename VMT::TV *) load_addr);

        T spill_left[N*2];
        T spill_right[N*2];

        T* RESTRICT spill_left_end = spill_left;
        // partition_block expects the left/right *write* pointers to point
        // to the next vector write position, for right write pointer
        // this means N elements BEFORE the end of the spill buffer
        T* RESTRICT spill_right_start = spill_right + N;
        T* RESTRICT spill_right_end = spill_right_start;

        memset(spill_left, 0x66, sizeof(spill_left));
        memset(spill_right, 0x66, sizeof(spill_right));

        PM::partition_block(data, PV, spill_left_end, spill_right_end);

        ASSERT_EQ(spill_left_end - spill_left, p);
        ASSERT_EQ(spill_right_start - spill_right_end, N - p);

        for (auto i = 0; i < p; ++i) {
            auto expected_value = fixture->get_expected_value(load_addr + i);
            ASSERT_EQ(spill_left[i], expected_value);
        }

        for (auto i = VMT::N - 1; i >= p; --i) {
            auto expected_value = fixture->get_expected_value(load_addr + i);
            ASSERT_EQ(spill_right_start[i], expected_value);

        }
    }
}

template <typename T, VM M>
void test_partition_alignment(PageWithLavaBoundariesFixture<T, M> *fixture)
{
    VXSORT_TEST_ISA();

    using VMT = vxsort::vxsort_machine_traits<T, M>;
    using PM = vxsort::partition_machine<T, M>;
    using AH = vxsort::alignment_hint<T, M>;
    static constexpr auto N = VMT::N;

    for (auto p = 0; p < VMT::N; p++) {
        auto * const left = fixture->page_with_data + p;
        auto * const right = fixture->page_with_data + fixture->num_elements - p;
        const auto pivot = fixture->get_expected_value(left + p) - 1;

        const auto PV = VMT::broadcast(pivot);

        AH align;
        align.calc_left_alignment(left);
        align.calc_right_alignment(right);

        T spill_left[N*2];
        T spill_right[N*2];

        T* RESTRICT spill_left_start = spill_left;
        T* RESTRICT spill_left_end = spill_left;

        // aligne_vectorized expects the left/right *write* pointers to point
        // to the boundary of the spill buffer, for right write pointer
        // this means the first element PAST the end of the spill buffer
        T* RESTRICT spill_right_start = spill_right + 2*N;
        T* RESTRICT spill_right_end = spill_right_start;

        memset(spill_left, 0x66, sizeof(spill_left));
        memset(spill_right, 0x66, sizeof(spill_right));

        auto left_masked_amount = align.left_masked_amount;
        auto right_unmasked_amount = align.right_unmasked_amount;

        T * RESTRICT left_next = left;
        T * RESTRICT right_next = right;

        //fmt::print("left_masked_amount: {}, right_unmasked_amount: {}\n", left_masked_amount, right_unmasked_amount);

        PM::align_vectorized(left_masked_amount, right_unmasked_amount,
                             PV,
                             left_next, right_next,
                             spill_left_start, spill_left_end,
                             spill_right_start, spill_right_end);

        // align vectorized API is build for continued
        // partitioning, so we need to update the right-pointing pointers
        // when vectorized partitioning is done by bumping them up by N elements
        right_next += N;
        spill_right_end += N;

        auto amount_read_left = left_next - left;
        auto amount_read_right = right - right_next;

        auto amount_partitioned_left = spill_left_end - spill_left_start;
        auto amount_partitioned_right = spill_right_start - spill_right_end;

        ASSERT_EQ(amount_partitioned_left + amount_partitioned_right,
                  amount_read_left + amount_read_right);

        ASSERT_EQ(spill_left_start - spill_left, align.left_masked_amount);

        for (auto i = 0; i < amount_partitioned_left; ++i) {
            ASSERT_LE(spill_left_start[i], pivot);
        }

        for (auto i = 0; i < amount_partitioned_right; ++i) {
            ASSERT_GT(spill_right_end[i], pivot);
        }
    }
}

};

#endif //VXSORT_PARTITION_MACHINE_TEST_H
