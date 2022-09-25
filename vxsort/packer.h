#ifndef VXSORT_PACKER_H
#define VXSORT_PACKER_H

#include <cstdint>
#include <limits>
#include <type_traits>
#include <cassert>
#include "defs.h"
#include "alignment.h"
#include "vector_machine/machine_traits.h"

#include <immintrin.h>
#include <cstdio>

namespace vxsort {

template<typename TFrom, typename TTo, vector_machine M, int Shift = 0, int Unroll = 1, int MinLength = 1>
class packer {
    static_assert(Shift <= 31, "Shift must be in the range 0..31");
    static_assert(Unroll >= 1, "Unroll can be in the range 1..4");
    static_assert(Unroll <= 4, "Unroll can be in the range 1..4");

    using VMT = vxsort_machine_traits<TFrom, M>;
    typedef typename VMT::TV TV;
    static const int N = sizeof(TV) / sizeof(TFrom);
    typedef alignment_hint<sizeof(TV)> AH;

    static const size_t ALIGN = AH::ALIGN;
    static const size_t ALIGN_MASK = ALIGN - 1;


    static INLINE TV pack_vectorized(const TV base_vec, TV d01, TV d02) {
        if (Shift > 0) { // This is statically compiled in/out
            d01 = VMT::shift_right(d01, Shift);
            d02 = VMT::shift_right(d02, Shift);
        }
        d01 = VMT::sub(d01, base_vec);
        d02 = VMT::sub(d02, base_vec);

        auto packed_data = VMT::pack_unordered(d01, d02);
        return packed_data;
    }

    static NOINLINE void unpack_vectorized(const TV baseVec, TV d01, TV& u01, TV& u02) {
        VMT::unpack_ordered(d01, u01, u02);

        u01 = VMT::add(u01, baseVec);
        u02 = VMT::add(u02, baseVec);

        if (Shift > 0) { // This is statically compiled in/out
            u01 = VMT::shift_left(u01, Shift);
            u02 = VMT::shift_left(u02, Shift);
        }
    }

public:

    static TTo *pack(TFrom *mem, std::size_t len, TFrom base) {
        TFrom offset = VMT::template shift_n_sub<Shift>(base, (TFrom) std::numeric_limits<TTo>::min());
        auto offset_vec = VMT::broadcast(offset);

        auto pre_aligned_mem = reinterpret_cast<TFrom *>(reinterpret_cast<size_t>(mem) & ~ALIGN_MASK);

        auto mem_read = mem;
        auto mem_write = (TTo *) mem;

        // Include a "special" pass to handle very short scalar
        // passes
        if (MinLength < N && len < N) {
            while (len--) {
                *(mem_write++) = (TTo)VMT::template  shift_n_sub<Shift>(*(mem_read++), offset);
            }
            return (TTo *) mem;
        }

        // We have at least
        // one vector worth of data to handle
        // Let's try to align to vector size first

        if (pre_aligned_mem < mem) {
            const auto alignment_point = pre_aligned_mem + N;
            len -= (alignment_point - mem_read);
            while (mem_read < alignment_point) {
                *(mem_write++) = (TTo)VMT::template shift_n_sub<Shift>(*(mem_read++), offset);
            }
        }

        assert(AH::is_aligned(mem_read));

        auto memv_read = (TV *) mem_read;
        auto memv_write = (TV *) mem_write;

        auto lenv = len / N;
        len -= (lenv * N);

        while (lenv >= 2 * Unroll) {
            assert(memv_read >= memv_write);

            TV d01, d02, d03, d04, d05, d06, d07, d08;

            do {
                d01 = VMT::load_vec(memv_read + 0);
                d02 = VMT::load_vec(memv_read + 1);
                if (Unroll == 1) break;
                d03 = VMT::load_vec(memv_read + 2);
                d04 = VMT::load_vec(memv_read + 3);
                if (Unroll == 2) break;
                d05 = VMT::load_vec(memv_read + 4);
                d06 = VMT::load_vec(memv_read + 5);
                if (Unroll == 3) break;
                d07 = VMT::load_vec(memv_read + 6);
                d08 = VMT::load_vec(memv_read + 7);
                break;
            } while (true);

            do {
                VMT::store_vec(memv_write + 0, pack_vectorized(offset_vec, d01, d02));
                if (Unroll == 1) break;
                VMT::store_vec(memv_write + 1, pack_vectorized(offset_vec, d03, d04));
                if (Unroll == 2) break;
                VMT::store_vec(memv_write + 2, pack_vectorized(offset_vec, d05, d06));
                if (Unroll == 3) break;
                VMT::store_vec(memv_write + 3, pack_vectorized(offset_vec, d07, d08));
                break;
            } while(true);

            memv_read += 2*Unroll;
            memv_write += Unroll;
            lenv -= 2*Unroll;
        }

        if (Unroll > 1) {
            while (lenv >= 2) {
                assert(memv_read >= memv_write);
                TV d01, d02;

                d01 = VMT::load_vec(memv_read + 0);
                d02 = VMT::load_vec(memv_read + 1);

                VMT::store_vec(memv_write + 0, pack_vectorized(offset_vec, d01, d02));
                memv_read += 2;
                memv_write++;
                lenv -= 2;
            }
        }

        len += lenv * N;

        mem_read = (TFrom *) memv_read;
        mem_write = (TTo *) memv_write;

        while (len-- > 0) {
            *(mem_write++) = (TTo)VMT::template shift_n_sub<Shift>(*(mem_read++), offset);
        }

        return (TTo *) mem;
    }


    static void unpack(TTo *mem, std::size_t len, TFrom base) {
        TFrom offset = VMT::template shift_n_sub<Shift>(base, (TFrom) std::numeric_limits<TTo>::min());
        auto offset_vec = VMT::broadcast(offset);

        auto mem_read = mem + len;
        auto mem_write = reinterpret_cast<TFrom*>(mem) + len;


        // Include a "special" pass to handle very short lengths
        if (MinLength < 2 * N && len < 2 * N) {
            while (len--) {
                *(--mem_write) = VMT::template unshift_and_add<Shift>(*(--mem_read), offset);
            }
            return;
        }

        auto pre_aligned_mem = reinterpret_cast<TTo *>(reinterpret_cast<size_t>(mem_read) & ~ALIGN_MASK);

        if (pre_aligned_mem < mem_read) {
            len -= (mem_read - pre_aligned_mem);
            while (mem_read > pre_aligned_mem) {
                *(--mem_write) = VMT::template unshift_and_add<Shift>(*(--mem_read), offset);
            }
        }

        assert(AH::is_aligned(mem_read));

        auto lenv = len / (N * 2);
        auto memv_read = ((TV *) mem_read) - 1;
        auto memv_write = ((TV *) mem_write) - 2;
        len -= lenv * N * 2;

        while (lenv >= Unroll) {
            assert(memv_read <= memv_write);

            TV d01, d02, d03, d04;
            TV u01, u02, u03, u04, u05, u06, u07, u08;

            do {
                d01 = VMT::load_vec(memv_read + 0);
                if (Unroll == 1) break;
                d02 = VMT::load_vec(memv_read - 1);
                if (Unroll == 2) break;
                d03 = VMT::load_vec(memv_read - 2);
                if (Unroll == 3) break;
                d04 = VMT::load_vec(memv_read - 3);
                break;
            } while(true);

            do {
                unpack_vectorized(offset_vec, d01, u01, u02);
                VMT::store_vec(memv_write + 0, u01);
                VMT::store_vec(memv_write + 1, u02);
                if (Unroll == 1) break;
                unpack_vectorized(offset_vec, d02, u03, u04);
                VMT::store_vec(memv_write - 2, u03);
                VMT::store_vec(memv_write - 1, u04);
                if (Unroll == 2) break;
                unpack_vectorized(offset_vec, d03, u05, u06);
                VMT::store_vec(memv_write - 4, u05);
                VMT::store_vec(memv_write - 3, u06);
                if (Unroll == 3) break;
                unpack_vectorized(offset_vec, d04, u07, u08);
                VMT::store_vec(memv_write - 6, u07);
                VMT::store_vec(memv_write - 5, u08);
                break;
            } while(true);

            memv_read -= Unroll;
            memv_write -= 2 * Unroll;
            lenv -= Unroll;
        }

        if (Unroll > 1) {
            while (lenv >= 1) {
                assert(memv_read <= memv_write);

                TV d01;
                TV u01, u02;

                d01 = VMT::load_vec(memv_read + 0);

                unpack_vectorized(offset_vec, d01, u01, u02);
                VMT::store_vec(memv_write + 0, u01);
                VMT::store_vec(memv_write + 1, u02);

                --memv_read;
                memv_write -= 2;
                --lenv;
            }
        }

        mem_read = (TTo *) (memv_read + 1);
        mem_write = (TFrom *) (memv_write + 2);

        while (len-- > 0) {
            *(--mem_write) = VMT::template unshift_and_add<Shift>(*(--mem_read), offset);
        }
    }

};

} // namespace vxsort

#endif  // VXSORT_PACKER_H
