#ifndef VXSORT_MINI_FIXTURES_H
#define VXSORT_MINI_FIXTURES_H

#include <gtest/gtest.h>
#ifndef _WIN32
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

static const i32 page_size = get_page_size();

template <typename T, VM M, bool for_packing = false>
class PageWithLavaBoundariesFixture : public ::testing::Test {
    using VMT = vxsort::vxsort_machine_traits<T, M>;
    static constexpr i32 N = VMT::N;
    static_assert(N < 256, "N must be < 256");

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
        for (usize i = 0; i < num_elements; i++) {
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
        static constexpr T max_value = for_packing ? (T)std::numeric_limits<typename VMT::TPACK>::max() : (T)std::numeric_limits<T>::max();
        for (auto n = 0; n < N; n++) {
            expected_values[n] = (T)n+1;
            ASSERT_LE(expected_values[n], max_value);
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

};

#endif //VXSORT_MINI_FIXTURES_H
