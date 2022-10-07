#ifndef VXSORT_MASKED_LOAD_STORE_TEST_H
#define VXSORT_MASKED_LOAD_STORE_TEST_H

#include <gtest/gtest.h>
#ifndef WIN32
#include <sys/mman.h>
#else
#ifndef NOMINMAX
# define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN 1
#include <Windows.h>
#endif

#include "defs.h"
#include "vector_machine/machine_traits.h"
#include "isa_detection.h"
#include "alignment.h"

namespace vxsort_tests {
using namespace vxsort::types;
using VM = vxsort::vector_machine;

static inline usize get_page_size()
{
    usize page_size;
#ifdef WIN32
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    page_size = sys_info.dwPageSize;
#else
    page_size = sysconf(_SC_PAGESIZE);
#endif
    return page_size;
}

static i32 page_size = get_page_size();

template <typename T, VM M>
class MaskedLoadStoreTest : public ::testing::Test {
    static constexpr u8 GARBAGE_BYTE = 0x66;
    using VMT = vxsort::vxsort_machine_traits<T, M>;
    static constexpr i32 N = VMT::N;

protected:

    u8 *create_mapping_with_boundary_pages() {
#ifdef WIN32
        auto *mem = (u8 *) VirtualAlloc(nullptr, 3*page_size, MEM_COMMIT, PAGE_READWRITE);
        DWORD old_protect;
        VirtualProtect(mem, page_size, PAGE_NOACCESS, &old_protect);
        VirtualProtect(mem + 2*page_size, page_size, PAGE_NOACCESS, &old_protect);
        return mem;
#else
        auto *mem = (u8 *) mmap(nullptr, 3*page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        // Make the first and last inaccessible
        mprotect(mem, page_size, PROT_NONE);
        mprotect(mem + 2*page_size, page_size, PROT_NONE);
        return mem;
#endif
    }

    void SetUp() override {
        // Map 3 pages
        mem = create_mapping_with_boundary_pages();
        generate_expected_values();
        num_elements = page_size / sizeof(T);

        page_with_data = reinterpret_cast<T *>(mem + page_size);
        for (auto i = 0; i < page_size / sizeof(T); i++) {
            auto *p = &page_with_data[i];
            *p = get_expected_value(p);
        }
    }

    void destroy_mapping() {
#ifdef WIN32
        VirtualFree(mem, 3*page_size, MEM_DECOMMIT);
#else
        munmap(mem, 3*page_size);
#endif

    }

    void TearDown() override {
        destroy_mapping();
    }

    void generate_expected_values()
    {
        for (auto n = 0; n < N; n++) {
            T expected_byte = n;
            T expected_value = expected_byte;

            for (auto i = 1; i < sizeof(T); i++)
                expected_value = expected_value << 8 | expected_byte;

            expected_values[n] = expected_value;
        }
    }

    T expected_values[N];
    u8 *mem;

public:
    T get_expected_value(const T *p)
    {
        const auto offset_in_elements = (((usize) p) / sizeof(T)) & (N-1);

        return expected_values[offset_in_elements];
    }


    T *page_with_data;
    usize num_elements;

};

template <typename T, VM M>
void test_prefix_mask_load_on_page_boundary(MaskedLoadStoreTest<T, M> *fixture)
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
void test_suffix_mask_load_on_page_boundary(MaskedLoadStoreTest<T, M> *fixture)
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
void test_left_alignment_and_masked_loads(MaskedLoadStoreTest<T, M> *fixture)
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
void test_right_alignment_and_masked_loads(MaskedLoadStoreTest<T, M> *fixture)
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
