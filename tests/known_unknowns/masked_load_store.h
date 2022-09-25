#ifndef VXSORT_MASKED_LOAD_STORE_TEST_H
#define VXSORT_MASKED_LOAD_STORE_TEST_H

#include <gtest/gtest.h>
#include <sys/mman.h>

#include "defs.h"
#include "vector_machine/machine_traits.h"
#include "vector_machine/machine_traits.avx2.h"
#include "vector_machine/machine_traits.avx512.h"
#include "isa_detection.h"

#include <fmt/format.h>

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

static i32 page_size = sysconf(_SC_PAGESIZE);


template <typename T>
class MaskedLoadStoreTest : public ::testing::Test {
    static constexpr u8 GARBAGE_BYTE = 0x66;

protected:
    void SetUp() override {
        // Map 3 pages
        mem = (u8 *) mmap(nullptr, 3*page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        // Make the first and last inaccessible
        mprotect(mem, page_size, PROT_NONE);
        mprotect(mem + 2*page_size, page_size, PROT_NONE);
        memset(mem + page_size, GARBAGE_BYTE, page_size);
        page_with_0x66 = reinterpret_cast<T *>(mem + page_size);
    }

    void TearDown() override {
        munmap(mem, 3*page_size);
    }

    T get_expected_value()
    {
        T expected_value = GARBAGE_BYTE;
        for (auto i = 0; i < sizeof(T); i++)
            expected_value = expected_value << 8 | GARBAGE_BYTE;
        return expected_value;
    }

    u8 *mem;
    T *page_with_0x66;
};

template <typename T, VM M>
void test_prefix_mask_load_on_page_boundary(T *page_with_0x66, i32 num_bytes, T expected_value)
{
    if (!::vxsort::supports_vector_machine(M)) {
        GTEST_SKIP_("Current CPU does not support the minimal features for this test");
        return;
    }

    using VMT = vxsort::vxsort_machine_traits<T, M>;
    static constexpr auto MAX = std::numeric_limits<T>::max();
    const auto MAXV = VMT::broadcast(MAX);

    const auto num_elements = num_bytes / sizeof(T);

    for (auto w = 1; w < VMT::N; w++) {
        auto mask = VMT::generate_prefix_mask(w);
        auto *load_offset = page_with_0x66 + num_elements - w;
        auto result = VMT::load_masked_vec((typename VMT::TV *) load_offset, MAXV, mask);
        auto &res_array = reinterpret_cast<T(&)[VMT::N]>(result);

        for (auto i = 0; i < w; ++i)
            EXPECT_EQ(res_array[i], expected_value);
        for (auto i = w; i < VMT::N; ++i)
            EXPECT_EQ(res_array[i], MAX);
    }
}

template <typename T, VM M>
void test_suffix_mask_load_on_page_boundary(T *page_with_0x66, i32 num_bytes, T expected_value)
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
        auto *load_offset = page_with_0x66 - VMT::N + w;
        auto result = VMT::load_masked_vec((typename VMT::TV *) load_offset, MAXV, mask);
        auto &res_array = reinterpret_cast<T(&)[VMT::N]>(result);

        for (auto i = 0; i < VMT::N - w; ++i)
            EXPECT_EQ(res_array[i], MAX);
        for (auto i = VMT::N - w; i < VMT::N; ++i)
            EXPECT_EQ(res_array[i], expected_value);
    }
}

};

#endif //VXSORT_MASKED_LOAD_STORE_TEST_H