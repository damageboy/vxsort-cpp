#ifndef BITONIC_SORT_H
#define BITONIC_SORT_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <cassert>
#include "../defs.h"
#include "vector_machine/machine_traits.h"
#include "bitonic_machine.h"

namespace vxsort {
namespace smallsort {
using namespace std;
using namespace vxsort::types;

#ifdef VXSORT_COMPILER_MSVC
#define __builtin_clzl __lzcnt64
#endif

template <typename T, vector_machine M>
struct bitonic {
    using BM = bitonic_machine<T, M>;
    using VMT = vxsort_machine_traits<T, M>;
    static constexpr T MAX = std::numeric_limits<T>::max();
    typedef typename VMT::TV TV;
    static constexpr i32 N = sizeof(TV) / sizeof(T);
    static_assert(is_powerof2(N), "vector-size / element-size must be a power of 2");

private:

    static INLINE u64 closest_pow2(size_t x) {
        return x == 1U ? 1U : 1U<<(64-__builtin_clzl(x-1));
    }

public:
    static void sort(T* RESTRICT ptr, usize length) {
    // We keep up to the last 4 vectors
    // in this temp space, because we need to deal with inputs < 4 vectors
    // and also deal with the tail of the array when it isn't exactly divisible by 4
    // vectors
    TV slack[4];

    // Full vectors
    auto v = length / N;
    // # elements in the last vector
    const auto remainder = static_cast<i32>(length - v * N);
    const auto slack_v = (v % 4) + (remainder ? 1 : 0);

    // Load/Store mask for the last vector
    const auto mask = VMT::generate_prefix_mask(remainder);
    // How many vectors in the last group of up to 4 vectors
    auto last_chunk_v = slack_v;

    auto * RESTRICT p_start = (TV*)ptr;
    const auto p_end_inplace = p_start + ((v/4) * 4);
    const auto p_virtual_end = p_end_inplace + ((slack_v > 0) ? 4 : 0);
    v += (remainder > 0) ? 1 : 0;

    auto p_exit_loop = p_end_inplace;

    TV * RESTRICT p = p_start;
    TV d01, d02, d03, d04;

    for (; p < p_exit_loop; p += 4) {
        d01 = VMT::load_vec(p + 0);
        d02 = VMT::load_vec(p + 1);
        d03 = VMT::load_vec(p + 2);
        d04 = VMT::load_vec(p + 3);
        ugly_hack_1:
        BM::sort_04v_ascending(d01, d02, d03, d04);
        VMT::store_vec(p + 0, d01);
        VMT::store_vec(p + 1, d02);
        VMT::store_vec(p + 2, d03);
        VMT::store_vec(p + 3, d04);
    }

    // We jump into the middle of the loop just for one last-time
    // to handle the schwanz
    p_exit_loop = nullptr;
    switch (last_chunk_v) {
        case 1:
            d04 = d03 = d02 = VMT::broadcast(MAX);
            d01 = VMT::load_partial_vec(p + 0, d02, mask);
            last_chunk_v = 0; p = slack; goto ugly_hack_1;
        case 2:
            d04 = d03 = VMT::broadcast(MAX);
            d01 = VMT::load_vec(p + 0);
            d02 = VMT::load_partial_vec(p + 1, d03, mask);
            last_chunk_v = 0; p = slack; goto ugly_hack_1;
        case 3:
            d04 = VMT::broadcast(MAX);
            d01 = VMT::load_vec(p + 0);
            d02 = VMT::load_vec(p + 1);
            d03 = VMT::load_partial_vec(p + 2, d04, mask);
            last_chunk_v = 0; p = slack; goto ugly_hack_1;
        case 4:
            d01 = VMT::load_vec(p + 0);
            d02 = VMT::load_vec(p + 1);
            d03 = VMT::load_vec(p + 2);
            d04 = VMT::load_partial_vec(p + 3, VMT::broadcast(MAX), mask);
            last_chunk_v = 0; p = slack; goto ugly_hack_1;
    }
    last_chunk_v = slack_v;
    p_exit_loop = p_end_inplace;

    TV* RESTRICT p1;
    TV* RESTRICT p2;
    TV *p2_end;
    i32 half_stride;

    const auto max_v = closest_pow2(v);
    for (u32 i = 8; i <= max_v; i *= 2) {
        for (p = p_start; p < p_virtual_end; p += i) {
            half_stride = i / 2;
            p1 = p + half_stride - 1;
            p2 = p + half_stride;
            p2_end = std::min(p + i, p_virtual_end);
            for (; p2 < p2_end; p1 -= 4, p2 += 4) {
                if (p2 >= p_end_inplace) {
                    p2 = slack;
                    p2_end = slack + 4;
                }

                TV dl, dr;

                dl = VMT::load_vec(p1 - 0);
                dr = VMT::load_vec(p2 + 0);
                BM::cross_min_max(dl, dr);
                VMT::store_vec(p1 - 0, dl);
                VMT::store_vec(p2 + 0, dr);

                dl = VMT::load_vec(p1 - 1);
                dr = VMT::load_vec(p2 + 1);
                BM::cross_min_max(dl, dr);
                VMT::store_vec(p1 - 1, dl);
                VMT::store_vec(p2 + 1, dr);

                dl = VMT::load_vec(p1 - 2);
                dr = VMT::load_vec(p2 + 2);
                BM::cross_min_max(dl, dr);
                VMT::store_vec(p1 - 2, dl);
                VMT::store_vec(p2 + 2, dr);

                dl = VMT::load_vec(p1 - 3);
                dr = VMT::load_vec(p2 + 3);
                BM::cross_min_max(dl, dr);
                VMT::store_vec(p1 - 3, dl);
                VMT::store_vec(p2 + 3, dr);
            }
        }

        const auto half_i = i /2;

        for (i32 j = half_i; j >= 8; j /= 2) {
            const auto half_stride = j/2;
            for (auto p = p_start; p < p_virtual_end; p += j) {
                p1 = p;
                p2 = p + half_stride;
                p2_end = std::min(p + j, p_virtual_end);
                for (; p2 < p2_end; p1 += 4, p2 += 4) {
                    if (p2 >= p_end_inplace) {
                        p2 = slack;
                        p2_end = slack + 4;
                    }

                    TV dl, dr;

                    dl = VMT::load_vec(p1 + 0);
                    dr = VMT::load_vec(p2 + 0);
                    BM::strided_min_max(dl, dr);
                    VMT::store_vec(p1 + 0, dl);
                    VMT::store_vec(p2 + 0, dr);

                    dl = VMT::load_vec(p1 + 1);
                    dr = VMT::load_vec(p2 + 1);
                    BM::strided_min_max(dl, dr);
                    VMT::store_vec(p1 + 1, dl);
                    VMT::store_vec(p2 + 1, dr);

                    dl = VMT::load_vec(p1 + 2);
                    dr = VMT::load_vec(p2 + 2);
                    BM::strided_min_max(dl, dr);
                    VMT::store_vec(p1 + 2, dl);
                    VMT::store_vec(p2 + 2, dr);

                    dl = VMT::load_vec(p1 + 3);
                    dr = VMT::load_vec(p2 + 3);
                    BM::strided_min_max(dl, dr);
                    VMT::store_vec(p1 + 3, dl);
                    VMT::store_vec(p2 + 3, dr);
                }
            }
        }
        p = p_start;
        for (; p < p_exit_loop; p += 4) {
            ugly_hack_4:
            d01 = VMT::load_vec(p + 0);
            d02 = VMT::load_vec(p + 1);
            d03 = VMT::load_vec(p + 2);
            d04 = VMT::load_vec(p + 3);
            BM::merge_04v_ascending(d01, d02, d03, d04);
            VMT::store_vec(p + 0, d01);
            VMT::store_vec(p + 1, d02);
            VMT::store_vec(p + 2, d03);
            VMT::store_vec(p + 3, d04);
        }
        if (last_chunk_v > 0) {
            p = slack;
            p_exit_loop = nullptr;
            last_chunk_v = 0;
            goto ugly_hack_4;
        }
        last_chunk_v = slack_v;
        p_exit_loop = p_end_inplace;

    }

    switch (last_chunk_v) {
        case 0:
            break;
        case 1:
            d01 = VMT::load_vec(slack + 0);
            VMT::store_masked_vec(p_end_inplace + 0, d01, mask);
        break;
        case 2:
            d01 = VMT::load_vec(slack + 0);
            d02 = VMT::load_vec(slack + 1);
            VMT::store_vec(p_end_inplace + 0, d01);
            VMT::store_masked_vec(p_end_inplace + 1, d02, mask);
        break;
        case 3:
            d01 = VMT::load_vec(slack + 0);
            d02 = VMT::load_vec(slack + 1);
            d03 = VMT::load_vec(slack + 2);
            VMT::store_vec(p_end_inplace + 0, d01);
            VMT::store_vec(p_end_inplace + 1, d02);
            VMT::store_masked_vec(p_end_inplace + 2, d03, mask);
        break;
        case 4:
            d01 = VMT::load_vec(slack + 0);
            d02 = VMT::load_vec(slack + 1);
            d03 = VMT::load_vec(slack + 2);
            d04 = VMT::load_vec(slack + 3);
            VMT::store_vec(p_end_inplace + 0, d01);
            VMT::store_vec(p_end_inplace + 1, d02);
            VMT::store_vec(p_end_inplace + 2, d03);
            VMT::store_masked_vec(p_end_inplace + 3, d04, mask);
        break;
    }
}
};
}  // namespace smallsort
}  // namespace vxsort
#endif
