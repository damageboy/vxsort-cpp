#ifndef VXSORT_PACK_MACHINE_TEST_H
#define VXSORT_PACK_MACHINE_TEST_H

#include <gtest/gtest.h>

#include <random>
#include <algorithm>
#include <span>
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
using PackMachineTest = PageWithLavaBoundariesFixture<T, M, true>;

template <typename T, VM M>
void test_packunpack(PackMachineTest<T, M> *fixture)
{
    if (!::vxsort::supports_vector_machine(M)) {
        GTEST_SKIP_("Current CPU does not support the minimal features for this test");
        return;
    }

    using VMT = vxsort::vxsort_machine_traits<T, M>;

    if (!VMT::supports_packing()) {
        GTEST_SKIP_("Current CPU does not support the minimal features for this test");
        return;
    }

    using PM = vxsort::pack_machine<T, M, 0>;
    static constexpr auto N = VMT::N;

    auto *load_addr = fixture->page_with_data;
    auto s = std::span(load_addr, N*2);
    const auto [min, max] = std::minmax_element(s.begin(), s.end());

    ASSERT_TRUE(VMT::template can_pack<0>(*max - *min));

    auto d1 = VMT::load_vec((typename VMT::TV *) load_addr);
    auto d2 = VMT::load_vec((typename VMT::TV *) load_addr + 1);

    auto constexpr MIN = T(std::numeric_limits<typename VMT::TPACK>::min());
    auto offset = VMT::template shift_n_sub<0>(*min, MIN);
    const auto offset_v = VMT::broadcast(offset);


    auto packed_v = PM::pack_vectors(d1, d2, offset_v);

    typename VMT::TV u1, u2;

    PM::unpack_vectors(offset_v, packed_v, u1, u2);

    T spill[N*2];
    VMT::store_vec((typename VMT::TV *) spill, u1);
    VMT::store_vec((typename VMT::TV *) spill+1, u2);

    std::vector<T> orig(s.begin(), s.end());
    for (auto u : spill) {
        auto it = std::find(orig.begin(), orig.end(), u);
        if (it == orig.end()) {
            GTEST_FAIL() << fmt::format("Expected to find unpacked value {} in {}", u, fmt::join(s, ", "));
        }
        orig.erase(it);
    }
    ASSERT_EQ(orig.size(), 0);
}

};

#endif //VXSORT_PACK_MACHINE_TEST_H
