#ifndef VXSORT_VXSORT_H
#define VXSORT_VXSORT_H

#ifdef __GNUC__
#ifdef __clang__
#pragma clang attribute push (__attribute__((target("popcnt"))), apply_to = any(function))
#else
#pragma GCC push_options
#pragma GCC target("popcnt")
#endif
#endif


#include <cassert>
#include <immintrin.h>

#include "alignment.h"
#include "defs.h"
#include "isa_detection.h"
#include "machine_traits.h"
#include "smallsort/bitonic_sort.h"
#include <algorithm>
#include <cstring>

#ifdef VXSORT_STATS
#include "stats/vxsort_stats.h"
#endif

namespace vxsort {
using namespace vxsort::types;

/**
 * sort primitives, quickly
 * @tparam T The primitive type being sorted
 * @tparam M The vector machine model/ISA (e.g. AVX2, AVX512 etc.)
 * @tparam Unroll The unroll factor, controls to some extent, the code-bloat/speedup ration at the call site
 *                Defaults to 1
 * @tparam Shift Optional; specify how many LSB bits are known to be zero in the original input. Can be used
 *               to further speed up sorting.
 */
template <typename T, vector_machine M, i32 Unroll=1, i32 Shift=0>
class vxsort {
    static_assert(Unroll >= 1, "Unroll can be in the range [1..12]");
    static_assert(Unroll <= 12, "Unroll can be in the range [1..12]");

private:
    using VMT = vxsort_machine_traits<T, M>;
    typedef typename VMT::TPACK TPACK;
    using VM_PACKED = vxsort_machine_traits<TPACK, M>;
    typedef typename VMT::TV TV;
    typedef alignment_hint<sizeof(TV)> AH;

    static const i32 ELEMENT_ALIGN = sizeof(T) - 1;
    static const i32 N = sizeof(TV) / sizeof(T);
    static_assert(is_powerof2(N), "vector-size / element-size must be a power of 2");

    static const i32 SMALL_SORT_THRESHOLD_ELEMENTS = 1024;
    static const i32 SMALL_SORT_THRESHOLD_VECTORS = SMALL_SORT_THRESHOLD_ELEMENTS / N;
    static const i32 SLACK_PER_SIDE_IN_VECTORS = Unroll;
    static const size_t ALIGN = AH::ALIGN;
    static const size_t ALIGN_MASK = ALIGN - 1;
    static const i32 SLACK_PER_SIDE_IN_ELEMENTS = SLACK_PER_SIDE_IN_VECTORS * N;
    // The formula for figuring out how much temporary space we need for partitioning:
    // 2 x the number of slack elements on each side for the purpose of partitioning in unrolled manner +
    // 2 x amount of maximal bytes needed for alignment (32)
    // one more vector's worth of elements since we write with N-way stores from both ends of the temporary area
    // and we must make sure we do not accidentally over-write from left -> right or vice-versa right on that edge...
    // In other words, while we allocated this much temp memory, the actual amount of elements inside said memory
    // is smaller by 8 elements + 1 for each alignment (max alignment is actually N-1, I just round up to N...)
    // This long sense just means that we over-allocate N+2 elements...
    static const i32 PARTITION_TMP_SIZE_IN_ELEMENTS =
            (2 * SLACK_PER_SIDE_IN_ELEMENTS + N + 4*N);

    static_assert(PARTITION_TMP_SIZE_IN_ELEMENTS < SMALL_SORT_THRESHOLD_ELEMENTS, "Unroll-level must match small-sorting threshold");
    static const i32 PackUnroll = (Unroll / 2 > 0) ? Unroll / 2 : 1;


    void reset(T* start, T* end) {
        _depth = 0;
        _start = start;
        _end = end;
    }

    T* _start = nullptr;
    T* _end = nullptr;

    T _temp[PARTITION_TMP_SIZE_IN_ELEMENTS];
    i32 _depth = 0;

    static i32 floor_log2_plus_one(size_t n) {
        auto result = 0;
        while (n >= 1) {
            result++;
            n /= 2;
        }
        return result;
    }
    static void swap(T* left, T* right) {
        auto tmp = *left;
        *left = *right;
        *right = tmp;
    }
    static void swap_if_greater(T* left, T* right) {
        if (*left <= *right)
            return;
        swap(left, right);
    }

    static void heap_sort(T* lo, T* hi) {
        size_t n = hi - lo + 1;
        for (auto i = n / 2; i >= 1; i--) {
            down_heap(i, n, lo);
        }
        for (size_t i = n; i > 1; i--) {
            swap(lo, lo + i - 1);
            down_heap(1, i - 1, lo);
        }
    }
    static void down_heap(size_t i, size_t n, T* lo) {
        auto d = *(lo + i - 1);
        while (i <= n / 2) {
            auto child = 2 * i;
            if (child < n && *(lo + child - 1) < (*(lo + child))) {
                child++;
            }
            if (!(d < *(lo + child - 1))) {
                break;
            }
            *(lo + i - 1) = *(lo + child - 1);
            i = child;
        }
        *(lo + i - 1) = d;
    }

    NOINLINE
    T* align_left_scalar_uncommon(T* read_left, T pivot,
                                  T*& tmp_left, T*& tmp_right) {
        if (((size_t)read_left & ALIGN_MASK) == 0)
            return read_left;

        auto* next_align = (T*)(((size_t)read_left + ALIGN) & ~ALIGN_MASK);
        while (read_left < next_align) {
            auto v = *(read_left++);
            if (v <= pivot) {
                *(tmp_left++) = v;
            } else {
                *(--tmp_right) = v;
            }
        }

        return read_left;
    }

    NOINLINE
    T* align_right_scalar_uncommon(T* read_right, T pivot,
                                   T*& tmp_left, T*& tmp_right) {
        if (((size_t)read_right & ALIGN_MASK) == 0)
            return read_right;

        auto* nextAlign = (T *) ((size_t)read_right & ~ALIGN_MASK);
        while (read_right > nextAlign) {
            auto v = *(--read_right);
            if (v <= pivot) {
                *(tmp_left++) = v;
            } else {
                *(--tmp_right) = v;
            }
        }

        return read_right;
    }

    void sort(T* left, T* right,
              T left_hint, T right_hint,
              AH realign_hint, i32 depth_limit) {
        auto length = static_cast<size_t>(right - left + 1);

        T* mid;
        switch (length) {
            case 0:
            case 1:
                return;
            case 2:
                swap_if_greater(left, right);
                return;
            case 3:
                mid = right - 1;
                swap_if_greater(left, mid);
                swap_if_greater(left, right);
                swap_if_greater(mid, right);
                return;
        }

        // Go to insertion sort below this threshold
        if (length <= SMALL_SORT_THRESHOLD_ELEMENTS) {
#ifdef VXSORT_STATS
            vxsort_stats<T>::bump_small_sorts();
            vxsort_stats<T>::record_small_sort_size(length);
#endif

            auto* const aligned_left = reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(left) & ~(N - 1));
            if (aligned_left < _start) {
                smallsort::bitonic<T, M>::sort(left, length);
                return;
            }

            length += (left - aligned_left);
            smallsort::bitonic<T, M>::sort(aligned_left, length);
            return;
        }

        // Detect a whole bunch of bad cases where partitioning
        // will not do well:
        // 1. Reverse sorted array
        // 2. High degree of repeated values (dutch flag problem, one value)
        if (depth_limit == 0) {
            heap_sort(left, right);
            _depth--;
            return;
        }

        depth_limit--;

        // This is going to be a bit weird:
        // Pre/Post alignment calculations happen here: we prepare hints to the
        // partition function of how much to align and in which direction
        // (pre/post). The motivation to do these calculations here and the actual
        // alignment inside the partitioning code is that here, we can cache those
        // calculations. As we recurse to the left we can reuse the left cached
        // calculation, And when we recurse to the right we reuse the right
        // calculation, so we can avoid re-calculating the same aligned addresses
        // throughout the recursion, at the cost of a minor code complexity

        // We use a long as a "struct" to pass on alignment hints to the
        // partitioning By packing 2 32 bit elements into it, as the JIT seem to not
        // do this. In reality, we need more like 2x 4bits for each side, but I
        // don't think there is a real difference'

        if (realign_hint.left_align == AH::REALIGN) {
            // Alignment flow:
            // * Calculate pre-alignment on the left
            // * See it would cause us an out-of bounds read
            // * Since we'd like to avoid that, we adjust for post-alignment
            // * No branches since we do branch->arithmetic
            auto* preAlignedLeft = reinterpret_cast<T*>(reinterpret_cast<size_t>(left) & ~ALIGN_MASK);
            auto cannotPreAlignLeft = (preAlignedLeft - _start) >> 63;
            realign_hint.left_align = (preAlignedLeft - left) + (N & cannotPreAlignLeft);
            assert(realign_hint.left_align >= -N && realign_hint.left_align <= N);
            assert(AH::is_aligned(left + realign_hint.left_align));
        }

        if (realign_hint.right_align == AH::REALIGN) {
            // Same as above, but in addition:
            // right is pointing just PAST the last element we intend to partition
            // (it's pointing to where we will store the pivot!) So we calculate alignment based on
            // right - 1
            auto* preAlignedRight = reinterpret_cast<T*>(((reinterpret_cast<size_t>(right) - 1) & ~ALIGN_MASK) + ALIGN);
            auto cannotPreAlignRight = (_end - preAlignedRight) >> 63;
            realign_hint.right_align = (preAlignedRight - right - (N & cannotPreAlignRight));
            assert(realign_hint.right_align >= -N && realign_hint.right_align <= N);
            assert(AH::is_aligned(right + realign_hint.right_align));
        }

        // Compute median-of-three, of:
        // the first, mid and one before last elements
        mid = left + ((right - left) / 2);
        swap_if_greater(left, mid);
        swap_if_greater(left, right - 1);
        swap_if_greater(mid, right - 1);

        // Pivot is mid, place it in the right-hand side
        swap(mid, right);


        // Some types (e.g. integers) can be potentially packed to a lower bit width while being partitioned.
        // Examples: i64 -> i32 -> i16, u32 -> u16, and so on...
        // For vxsort to take advantage of this, it needs to know what is the maximal/minimal value
        // that it will encounter within a given partition. If the span between that min/max is smaller-or-equal to
        // the largest value that can be expressed by the next smaller bit-width, we're good to go.
        //
        // Example: If we are dealing with i64 types, but somehow know the min/max span of a given partition happens
        // to be <= 2^32, this means we can pack the entire partition to take 1/2 of it's space in memory/cache/vector-units
        // terms and proceed to sort it with the new, more efficient bit-width.
        //
        // If we think of this naively, we might conclude there is NO easy way of knowing the min/max range without having
        // the *caller* of the sort function provide the min/max values.
        // While that is the most preferable method, that allows for earlier packing throughout the sorting effort
        // we can also pull this off without such hints.
        // *Any* internal partition, is by definition, surrounded by already existing elements of the original sorting problem
        // that are *guaranteed*, given the divide-conquer nature of quick-sort , to be smaller/larger than the partition we are
        // tasked with partitioning around the pivot. As such, any such values from over the edge of the existing partition can
        // serve as a min/max boundary for the aforementioned optimization.
        // While the values sampled from the exterior of an internal partition will not represent the TRUE min/max of the current
        // partition, they might still, statistically speaking, provide an opportunity to pack while sorting the data.
        if (VMT::type_supports_packing()) {
            if (VMT::template can_pack<Shift>(right_hint - left_hint)) {
#ifdef VXSORT_STATS
                vxsort_stats<T>::bump_packs(length);
                vxsort_stats<T>::bump_unpacks(length);
#endif
                auto left_length = vectorized_packed_partition(left, right, left_hint, realign_hint);
                auto right_length = length - left_length;
                auto* const left_packed_start  = reinterpret_cast<TPACK *>(left);
                auto* const left_packed_end    = left_packed_start + left_length - 1;
                auto* const right_packed_end   = reinterpret_cast<TPACK *>(right + 1) - 1;
                auto* const right_packed_start = right_packed_end - right_length + 1;

                auto packed_sorter = vxsort<TPACK, M, Unroll>();
                packed_sorter.sort(left_packed_start, left_packed_end);
                packed_sorter.sort(right_packed_start, right_packed_end);

                vectorized_unpack_backward<2>(left_packed_start, left_length, left_hint);
                vectorized_unpack_forward<2>(right_packed_end + 1, right_length, left_hint);
                return;
            }
        }

        auto sep = vectorized_partition<Unroll>(left, right, realign_hint);

        _depth++;
        sort(left, sep - 2, left_hint, *sep, realign_hint.realign_right(), depth_limit);
        sort(sep, right, *(sep - 2), right_hint, realign_hint.realign_left(), depth_limit);
        _depth--;
    }

    template <typename TFriend, vector_machine MFriend, i32 UnrollFriend, i32 ShiftFriend>
    friend class vxsort;

    static INLINE void partition_block(TV& data_vec, const TV P,
                                       T*& left, T*& right)
    {
#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_vec_loads();
        vxsort_stats<T>::bump_vec_stores(2);
#endif
      if (VMT::supports_compress_writes()) {
        partition_block_with_compress(data_vec, P, left, right);
      } else {
        partition_block_without_compress(data_vec, P, left, right);
      }
    }

    static INLINE void partition_block_without_compress(TV& data_vec, const TV P,
                                                        T*& left, T*& right)
    {
#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_perms();
#endif
        auto mask = VMT::get_cmpgt_mask(data_vec, P);
        data_vec = VMT::partition_vector(data_vec, mask);
        VMT::store_vec(reinterpret_cast<TV*>(left), data_vec);
        VMT::store_vec(reinterpret_cast<TV*>(right), data_vec);
        auto popcnt = -_mm_popcnt_u64(mask);
        right += popcnt;
        left += popcnt + N;
    }

    static INLINE void partition_block_with_compress(TV& data_vec, const TV P,
                                                     T*& left, T*& right)
    {
        auto mask = VMT::get_cmpgt_mask(data_vec, P);
        auto popcnt = -_mm_popcnt_u64(mask);
        VMT::store_compress_vec(reinterpret_cast<TV*>(left), data_vec, ~mask);
        VMT::store_compress_vec(reinterpret_cast<TV*>(right + N + popcnt), data_vec, mask);
        right += popcnt;
        left += popcnt + N;
    }

    /// vectorized_partition - partition an array with vector operations
    /// \tparam InnerUnroll - The amount of unrolling / code-bloat
    /// \param left - pointer (inclusive) to the left edge of the partition
    /// \param right - pointer (inclusive) to the right edge of the partition.
    ///                Note: as part of the internal convention, this points
    ///                      to where the pivot for this call is stored.
    /// \param hint - a (partially) cache hint used to communicate where the
    ///               the nearest vector-alignment left+right of the partition
    ///               is situated.
    /// \return A pointer to the new location
    /// Vectorized double-pumped (dual-sided) partitioning:
    /// We start with reading the pivot value passed on to use at the position
    /// pointed to by the 'right' parameter. We process the array from both ends.
    ///
    /// To get this rolling, we first read <InnerUnroll> x vector-sized elements
    /// from the left and right, each. This data is partitioned and stored in a
    /// small temporary buffer to make some room for the main block where an
    /// inplace partitioning loop is performed.
    ///Conceptually, the bulk
    /// of the processing looks like this after clearing out some initial space
    /// as described above:
    /// |-- InnerUnroll x -|                                    |-- InnerUnroll x -|
    ///        vector-size                                             vector-size
    //           bytes                                                   bytes
    /// [..........................................................................]
    //  ^wl                ^rl                                rr^                wr^
    /// Where:
    /// wl = write_left_v
    /// rl = read_left_v
    /// rr = read_right_v
    /// wr = write_right_v
    /// In every iteration, we select what side to read from based on how much
    /// space is left between head read/write pointer on each side...
    /// We read from where there is a smaller gap, e.g. that side
    /// that is closer to the unfortunate possibility of its write head
    /// overwriting its read head... By reading from THAT sides read "head",
    /// we're ensuring this does not happen.
    /// Each partitioning operation ends up reading from one of the side "heads"
    /// and distributing the partitioned values to both write "heads" according
    /// to how the data ends up being divvied up.
    /// A large chunk of code takes care of the issue of initially aligning
    /// the read pointers in such a way that all reads use pointers aligned to
    /// a vector unit, which greatly reduces the amount of execution resources
    /// required by a modern processor to read the required data.
    template<int InnerUnroll>
    T* vectorized_partition(T* const left, T* const right, const AH hint)
    {
        assert(right - left >= SMALL_SORT_THRESHOLD_ELEMENTS);
        assert((reinterpret_cast<size_t>(left) & ELEMENT_ALIGN) == 0);
        assert((reinterpret_cast<size_t>(right) & ELEMENT_ALIGN) == 0);

#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_partitions((right - left) + 1);
#endif

        // An additional unfortunate complexity we need to deal with is that the
        // right pointer must be decremented by another Vector256<T>.Count elements
        // Since the Load/Store primitives obviously accept start addresses
        auto pivot = *right;
        // We do this here just in case we need to pre-align to the right
        // We end up
        *right = std::numeric_limits<T>::max();

        // Broadcast the selected pivot
        const auto P = VMT::broadcast(pivot);

        auto read_left = left;
        auto read_right = right;

        auto tmp_start_left = _temp;
        auto tmp_left = tmp_start_left;
        auto tmp_start_right = _temp + PARTITION_TMP_SIZE_IN_ELEMENTS;
        auto tmp_right = tmp_start_right;

        tmp_right -= N;

        // the read heads always advance by 8 elements, or 32 bytes,
        // We can spend some extra time here to align the pointers
        // so they start at a cache-line boundary
        // Once that happens, we can read with Avx.LoadAlignedVector256
        // And also know for sure that our reads will never cross cache-lines
        // Otherwise, 50% of our AVX2 Loads will need to read from two cache-lines
        align_vectorized(left, right, hint, P, read_left, read_right, tmp_start_left, tmp_left, tmp_start_right, tmp_right);

        const auto leftAlign = hint.left_align;
        const auto rightAlign = hint.right_align;
        if (leftAlign > 0) {
            tmp_right += N;
            read_left = align_left_scalar_uncommon(read_left, pivot, tmp_left, tmp_right);
            tmp_right -= N;
        }

        if (rightAlign < 0) {
            tmp_right += N;
            read_right =
                    align_right_scalar_uncommon(read_right, pivot, tmp_left, tmp_right);
            tmp_right -= N;
        }

        assert(((size_t)read_left & ALIGN_MASK) == 0);
        assert(((size_t)read_right & ALIGN_MASK) == 0);

        assert((((size_t)read_right - (size_t)read_left) % ALIGN) == 0);
        assert((read_right - read_left) >= InnerUnroll * 2);

        // From now on, we are fully aligned
        // and all reading is done in full vector units
        auto read_left_v = (TV*)read_left;
        auto read_right_v = (TV*)read_right;
#ifndef NDEBUG
        read_left = nullptr;
        read_right = nullptr;
#endif

        for (auto u = 0; u < InnerUnroll; u++) {
            auto dl = VMT::load_vec(read_left_v + u);
            auto dr = VMT::load_vec(read_right_v - (u + 1));
            partition_block(dl, P, tmp_left, tmp_right);
            partition_block(dr, P, tmp_left, tmp_right);
        }

        // Fix-up tmp_right after the last vector operation
        // potentially *writing* through it is done
        tmp_right += N;
        // Adjust for the reading that was made above
        read_left_v += InnerUnroll;
        read_right_v -= InnerUnroll*2;
        TV* nextPtr;

        auto write_left = left;
        auto write_right = right - N;

        while (read_left_v < read_right_v) {
            if (write_right - ((T *)read_right_v) < (2 * (InnerUnroll * N) - N)) {
                nextPtr = read_right_v;
                read_right_v -= InnerUnroll;
            } else {
                // This is relevant for MSVC to not kill our perf
                mess_up_cmov();
                nextPtr = read_left_v;
                read_left_v += InnerUnroll;
            }

            TV d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12;

            switch (InnerUnroll) {
                case 12: d12 = VMT::load_vec(nextPtr + InnerUnroll - 12); [[fallthrough]];
                case 11: d11 = VMT::load_vec(nextPtr + InnerUnroll - 11); [[fallthrough]];
                case 10: d10 = VMT::load_vec(nextPtr + InnerUnroll - 10); [[fallthrough]];
                case  9: d09 = VMT::load_vec(nextPtr + InnerUnroll -  9); [[fallthrough]];
                case  8: d08 = VMT::load_vec(nextPtr + InnerUnroll -  8); [[fallthrough]];
                case  7: d07 = VMT::load_vec(nextPtr + InnerUnroll -  7); [[fallthrough]];
                case  6: d06 = VMT::load_vec(nextPtr + InnerUnroll -  6); [[fallthrough]];
                case  5: d05 = VMT::load_vec(nextPtr + InnerUnroll -  5); [[fallthrough]];
                case  4: d04 = VMT::load_vec(nextPtr + InnerUnroll -  4); [[fallthrough]];
                case  3: d03 = VMT::load_vec(nextPtr + InnerUnroll -  3); [[fallthrough]];
                case  2: d02 = VMT::load_vec(nextPtr + InnerUnroll -  2); [[fallthrough]];
                case  1: d01 = VMT::load_vec(nextPtr + InnerUnroll -  1);
            }

            switch (InnerUnroll) {
                case 12: partition_block(d12, P, write_left, write_right); [[fallthrough]];
                case 11: partition_block(d11, P, write_left, write_right); [[fallthrough]];
                case 10: partition_block(d10, P, write_left, write_right); [[fallthrough]];
                case  9: partition_block(d09, P, write_left, write_right); [[fallthrough]];
                case  8: partition_block(d08, P, write_left, write_right); [[fallthrough]];
                case  7: partition_block(d07, P, write_left, write_right); [[fallthrough]];
                case  6: partition_block(d06, P, write_left, write_right); [[fallthrough]];
                case  5: partition_block(d05, P, write_left, write_right); [[fallthrough]];
                case  4: partition_block(d04, P, write_left, write_right); [[fallthrough]];
                case  3: partition_block(d03, P, write_left, write_right); [[fallthrough]];
                case  2: partition_block(d02, P, write_left, write_right); [[fallthrough]];
                case  1: partition_block(d01, P, write_left, write_right);
            }
        }

        read_right_v += (InnerUnroll - 1);

        while (read_left_v <= read_right_v) {
            if (write_right - (T *)read_right_v < N) {
                nextPtr = read_right_v;
                read_right_v -= 1;
            } else {
                mess_up_cmov();
                nextPtr = read_left_v;
                read_left_v += 1;
            }

            auto d = VMT::load_vec(nextPtr);
            partition_block(d, P, write_left, write_right);
        }

        // 3. Copy-back the 4 registers + remainder we partitioned in the beginning
        auto left_tmp_size = tmp_left - tmp_start_left;
        memcpy(write_left, tmp_start_left, left_tmp_size * sizeof(T));
        write_left += left_tmp_size;
        auto right_tmp_size = tmp_start_right - tmp_right;
        memcpy(write_left, tmp_right, right_tmp_size * sizeof(T));

        // Shove to pivot back to the boundary
        *right = *write_left;
        *write_left++ = pivot;

        assert(write_left > left);
        assert(write_left <= right+1);

        return write_left;
    }

    /// vectorized_packed_partition - partition an array while performing a bit-width
    ///                               reducing pack operation to compress the data on
    ///                               the fly
    /// \param left - pointer (inclusive) to the left edge of the partition
    /// \param right - pointer (inclusive) to the right edge of the partition.
    ///                Note: as part of the internal convention, this points
    ///                      to where the pivot for this call is stored.
    /// \param min_bounding - the base value to subtract from each value before performing
    ///               the packing
    /// \param hint - a (partially) cache hint used to communicate where the
    ///               the nearest vector-alignment left+right of the partition
    ///               is situated.
    /// \return The amount of elements partitioned to the left side
    size_t vectorized_packed_partition(T* const left, T* const right, T min_bounding, const AH hint) {
        assert(right - left >= SMALL_SORT_THRESHOLD_ELEMENTS);
        assert((reinterpret_cast<size_t>(left) & ELEMENT_ALIGN) == 0);
        assert((reinterpret_cast<size_t>(right) & ELEMENT_ALIGN) == 0);

#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_partitions((right - left) + 1);
#endif

        auto pivot = *right;
        *right = std::numeric_limits<T>::max();

        // Broadcast the selected pivot
        const auto P = VMT::broadcast(pivot);
        // Create a vectorized version of the offset by which we need to
        // correct the data before packing it
        auto constexpr MIN = T(std::numeric_limits<TPACK>::min());
        auto offset = VMT::template shift_n_sub<Shift>(min_bounding, MIN);
        const TV offset_v = VMT::broadcast(offset);

        auto* read_left = left;
        auto* read_right = right;

        auto* tmp_start_left = _temp;
        auto* tmp_left = tmp_start_left;
        auto* tmp_start_right = _temp + PARTITION_TMP_SIZE_IN_ELEMENTS;
        auto* tmp_right = tmp_start_right;

        tmp_right -= N;

        // We take some extra time to attempt to read more than the exact partition extents
        // there-by killing off two birds: We aligh the read-pointers to vector-size units,
        // which greatly reduces the amount of split-cache-line load operations on the one hand.
        // At the same time, if we manage to accomplish this in a vectorized manner, we also
        // avoid all scalar work that may have been required to take care of the tail of partition
        // when that is not vector-sized.
        align_vectorized(left, right, hint, P, read_left, read_right, tmp_start_left, tmp_left, tmp_start_right, tmp_right);

        const auto left_align = hint.left_align;
        const auto right_align = hint.right_align;
        if (left_align > 0) {
            tmp_right += N;
            read_left = align_left_scalar_uncommon(read_left, pivot, tmp_left, tmp_right);
            tmp_right -= N;
        }

        if (right_align < 0) {
            tmp_right += N;
            read_right = align_right_scalar_uncommon(read_right, pivot, tmp_left, tmp_right);
            tmp_right -= N;
        }

        assert(((size_t)read_left & ALIGN_MASK) == 0);
        assert(((size_t)read_right & ALIGN_MASK) == 0);

        assert((((size_t)read_right - (size_t)read_left) % ALIGN) == 0);
        //assert((readRight - readLeft) >= InnerUnroll * 2);

        // From now on, we are fully aligned
        // and all reading is done in full vector units
        auto read_left_v = reinterpret_cast<TV*>(read_left);
        auto read_right_v = reinterpret_cast<TV*>(read_right);
#ifndef NDEBUG
        read_left = nullptr;
        read_right = nullptr;
#endif

        auto* write_left = reinterpret_cast<TPACK*>(left);
        auto* write_right = reinterpret_cast<TPACK*>(right+1) - 2*N;

        // We will be packing before partitioning, so
        // We must generate a pre-packed pivot
        const auto packed_pivot = VMT::template shift_n_sub<Shift>(pivot, offset);
        const TV PPP = VM_PACKED::broadcast(static_cast<TPACK>(packed_pivot));

        auto len_v = read_right_v - read_left_v;
        auto len_dv = len_v / 2;

        len_v -= len_dv * 2;

#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_vec_loads(2 * len_dv);
        vxsort_stats<T>::bump_vec_stores(2 * len_dv);
#endif

        for (auto i = 0; i < len_dv; i++) {
            auto dl = VMT::load_vec(read_left_v + i);
            auto dr = VMT::load_vec(read_right_v - (i + 1));

            // This is statically compiled in/out
            if (Shift > 0) {
                dl = VMT::shift_right(dl, Shift);
                dr = VMT::shift_right(dr, Shift);
            }
            dl = VMT::sub(dl, offset_v);
            dr = VMT::sub(dr, offset_v);

            auto packed_data = VMT::pack_unordered(dl, dr);

            vxsort<TPACK, M, Unroll>::partition_block(packed_data, PPP, write_left, write_right);
        }

        // We might have one more vector worth of stuff to partition, so we'll do it with
        // scalar partitioning into the tmp space
        if (len_v > 0) {
            auto slack = VMT::load_vec((TV *) (read_left_v + len_dv));
            partition_block(slack, P, tmp_left, tmp_right);
        }
        // Fix-up tmp_right after the last vector operation
        // potentially *writing* through it is done
        tmp_right += N;

        write_right += 2*N;

        for (auto *p = tmp_start_left; p < tmp_left; p++) {
            *(write_left++) = static_cast<TPACK>(VMT::template shift_n_sub<Shift>(*p, offset));
        }

        for (auto *p = tmp_right; p < tmp_start_right; p++) {
            *(--write_right) = static_cast<TPACK>(VMT::template shift_n_sub<Shift>(*p, offset));
        }

        *write_left++ = static_cast<TPACK>(packed_pivot);

        return write_left - reinterpret_cast<TPACK*>(left);
    }

    enum UnpackDirection {
        Left = -1,
        None = 0,
        Right = 1,
    };

    static NOINLINE void unpack_vectorized(const TV baseVec, TV d01, TV& u01, TV& u02) {
        VMT::unpack_ordered(d01, u01, u02);

        u01 = VMT::add(u01, baseVec);
        u02 = VMT::add(u02, baseVec);

        if (Shift > 0) { // This is statically compiled in/out
            u01 = VMT::shift_left(u01, Shift);
            u02 = VMT::shift_left(u02, Shift);
        }
    }

    template <typename TPTR = char>
    static TPTR* align_up(void* p, const intptr_t mask) {
        auto value = reinterpret_cast<intptr_t>(p);
        value += (-value) & mask;
        return reinterpret_cast<TPTR*>(value);
    }

    template <typename TPTR = char>
    static TPTR* align_down(void* p, const intptr_t mask) {
        auto value = reinterpret_cast<intptr_t>(p);
        value = value & ~mask;
        return reinterpret_cast<TPTR*>(value);
    }

    /// \brief Unpack a sequence of packed values into a sequence of unpacked values.
    ///        This implementation goes through the memory going backwards
    ///        (e.g. decreasing addresses) as depicted below:
    ///                length                                          x    - unreadable/writeable memory
    ///                   ▲                                            r/R  - readable elements
    ///                   │                                            ww/WW- written elements
    ///    ┌──────────────┴─────────────┐
    ///    ├────────────────────────────┴─────────────────────────────────────────────────────────────────┐
    ///    rRrRrRrRrRrRrRrRrRrRrRrRrRrRrRxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ///    ├────────────────────────────▲─────────────────────────────────────────────────────────────────┘
    ///    │                            │
    ///    │                            Reading
    ///    ▼          Direction         Starts
    /// mem_end    ◄──────────────      Here                          Writing
    ///    ▲              of                                          Starts
    ///    │           Unpacking                                      Here
    ///    │                                                          │
    ///    ├──────────────────────────────────────────────────────────▼───────────────────────────────────┐
    ///    wwWWwwWWwwWWwwWWwwWWwwWWwwWWwwWWwwWWwwWWwwWWwwWWwwWWwwWWwwWWxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ///    └──────────────────────────────────────────────────────────────────────────────────────────────┘

    /// \param mem_end
    /// \param len
    /// \param base
    template<int UnpackUnroll>
    void vectorized_unpack_backward(TPACK* const mem_end, size_t len, T base) {
        auto constexpr MIN = T(std::numeric_limits<TPACK>::min());
        T offset = VMT::template shift_n_sub<Shift>(base, MIN);

        auto mem_read = mem_end + len;
        auto mem_write = reinterpret_cast<T*>(mem_end) + len;

        // Include a "special" pass to handle very short lengths
        if (len < 2 * N) {
            while (len--) {
                *(--mem_write) = VMT::template unshift_and_add<Shift>(*(--mem_read), offset);
            }
            return;
        }

        auto align_point = align_down<TPACK>(mem_read, ALIGN_MASK);

        if (align_point < mem_read) {
            len -= (mem_read - align_point);
            while (mem_read > align_point) {
                *(--mem_write) = VMT::template unshift_and_add<Shift>(*(--mem_read), offset);
            }
        }

        assert(AH::is_aligned(mem_read));
        auto base_v = VMT::broadcast(offset);
        auto lenv = len / VM_PACKED::N;
        auto memv_read = reinterpret_cast<TV*>(mem_read) - 1;
        auto memv_write = reinterpret_cast<TV*>(mem_write) - 2;
        len -= lenv * N * 2;

        while (lenv >= UnpackUnroll) {
            assert(memv_read <= memv_write);

            TV d01, d02, d03, d04;
            do {
                d01 = VMT::load_vec(memv_read + 0);
                if (UnpackUnroll == 1) break;
                d02 = VMT::load_vec(memv_read - 1);
                if (UnpackUnroll == 2) break;
                d03 = VMT::load_vec(memv_read - 2);
                if (UnpackUnroll == 3) break;
                d04 = VMT::load_vec(memv_read - 3);
                break;
            } while(true);

            do {
                TV u01, u02, u03, u04, u05, u06, u07, u08;
                unpack_vectorized(base_v, d01, u01, u02);
                VMT::store_vec(memv_write + 0, u01);
                VMT::store_vec(memv_write + 1, u02);
                if (UnpackUnroll == 1) break;
                unpack_vectorized(base_v, d02, u03, u04);
                VMT::store_vec(memv_write - 2, u03);
                VMT::store_vec(memv_write - 1, u04);
                if (UnpackUnroll == 2) break;
                unpack_vectorized(base_v, d03, u05, u06);
                VMT::store_vec(memv_write - 4, u05);
                VMT::store_vec(memv_write - 3, u06);
                if (UnpackUnroll == 3) break;
                unpack_vectorized(base_v, d04, u07, u08);
                VMT::store_vec(memv_write - 6, u07);
                VMT::store_vec(memv_write - 5, u08);
                break;
            } while(true);

            memv_read -= UnpackUnroll;
            memv_write -= 2 * UnpackUnroll;
            lenv -= UnpackUnroll;
        }

        if (UnpackUnroll > 1) {
            while (lenv >= 1) {
                assert(memv_read <= memv_write);

                TV d01;
                TV u01, u02;

                d01 = VMT::load_vec(memv_read);

                unpack_vectorized(base_v, d01, u01, u02);
                VMT::store_vec(memv_write + 0, u01);
                VMT::store_vec(memv_write + 1, u02);

                --memv_read;
                memv_write -= 2;
                --lenv;
            }
        }

        mem_read = reinterpret_cast<TPACK*>(memv_read + 1);
        mem_write = reinterpret_cast<T*>(memv_write + 2);

        while (len-- > 0) {
            *(--mem_write) = VMT::template unshift_and_add<Shift>(*(--mem_read), offset);
        }
        assert(mem_read == reinterpret_cast<TPACK *>(mem_write));
        assert(mem_read == mem_end);
    }


    /// \brief Unpack a sequence of packed values into a sequence of unpacked values.
    ///        This implementation goes through the memory going forward
    ///        (e.g. increasing addresses) as depicted below:
    ///                                                                              length
    ///  x    - unreadable/writeable memory                                             ▲
    ///  r/R  - readable elements                                                       │
    ///  ww/WW- written elements                                         ┌──────────────┴──────────────┐
    /// ┌────────────────────────────────────────────────────────────────┴─────────────────────────────┤
    /// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxrRrRrRrRrRrRrRrRrRrRrRrRrRrRrRx
    /// └────────────────────────────────────────────────────────────────▲─────────────────────────────┤
    ///                                                                  │                             │
    ///                                                            Reading                             │
    ///                                                             Starts         Direction           ▼
    ///                                    Writing                    Here    ─────────────────►   mem_end
    ///                                    Starts                                      of              ▲
    ///                                    Here                                     Unpacking          │
    ///                                    │                                                           │
    /// ┌──────────────────────────────────▼───────────────────────────────────────────────────────────┤
    /// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxwwWWwwWWwwWWwwWWwwWWwwWWwwWWwwWWwwWWwwWWwwWWwwWWwwWWwwWWwwWWx
    /// └──────────────────────────────────────────────────────────────────────────────────────────────┘
    /// \tparam UnpackUnroll
    /// \param mem_end - pointer to the first data element *past* the last
    ///                  element to unpack(!)
    /// \param len - amount of elements to unpack
    /// \param base - the integer offset to add to each unpacked value
    ///               (for the purpose of unpacking)
    template<int UnpackUnroll>
    void vectorized_unpack_forward(TPACK* const mem_end, size_t len, T base) {
        auto constexpr MIN = T(std::numeric_limits<TPACK>::min());
        T offset = VMT::template shift_n_sub<Shift>(base, MIN);

        auto mem_read = mem_end - len;
        auto mem_write = reinterpret_cast<T*>(mem_end)  - len;

        // Include a "special" pass to handle very short lengths
        if (len < 2 * N) {
            while (len--) {
                *(mem_write++) = VMT::template unshift_and_add<Shift>(*(mem_read++), offset);
            }
            return;
        }

        auto align_point = align_up<TPACK>(mem_read, ALIGN_MASK);

        if (align_point > mem_read) {
            len -= (align_point - mem_read);
            while (mem_read < align_point) {
                *(mem_write++) = VMT::template unshift_and_add<Shift>(*(mem_read++), offset);
            }
        }

        assert(AH::is_aligned(mem_read));
        auto base_v = VMT::broadcast(offset);
        auto lenv = len / VM_PACKED::N;
        auto memv_read = reinterpret_cast<TV*>(mem_read);
        auto memv_write = reinterpret_cast<TV*>(mem_write);
        len -= lenv * N * 2;

        while (lenv >= UnpackUnroll) {
            assert(memv_read >= memv_write);

            TV d01, d02, d03, d04;
            do {
                d01 = VMT::load_vec(memv_read + 0);
                if (UnpackUnroll == 1) break;
                d02 = VMT::load_vec(memv_read + 1);
                if (UnpackUnroll == 2) break;
                d03 = VMT::load_vec(memv_read + 2);
                if (UnpackUnroll == 3) break;
                d04 = VMT::load_vec(memv_read + 3);
                break;
            } while(true);

            do {
                TV u01, u02, u03, u04, u05, u06, u07, u08;
                unpack_vectorized(base_v, d01, u01, u02);
                VMT::store_vec(memv_write + 0, u01);
                VMT::store_vec(memv_write + 1, u02);
                if (UnpackUnroll == 1) break;
                unpack_vectorized(base_v, d02, u03, u04);
                VMT::store_vec(memv_write + 2, u03);
                VMT::store_vec(memv_write + 3, u04);
                if (UnpackUnroll == 2) break;
                unpack_vectorized(base_v, d03, u05, u06);
                VMT::store_vec(memv_write + 4, u05);
                VMT::store_vec(memv_write + 5, u06);
                if (UnpackUnroll == 3) break;
                unpack_vectorized(base_v, d04, u07, u08);
                VMT::store_vec(memv_write + 6, u07);
                VMT::store_vec(memv_write + 7, u08);
                break;
            } while(true);

            memv_read += UnpackUnroll;
            memv_write += 2 * UnpackUnroll;
            lenv -= UnpackUnroll;
        }

        if (UnpackUnroll > 1) {
            while (lenv >= 1) {
                assert(memv_read >= memv_write);

                TV d01;
                TV u01, u02;

                d01 = VMT::load_vec(memv_read);

                unpack_vectorized(base_v, d01, u01, u02);
                VMT::store_vec(memv_write + 0, u01);
                VMT::store_vec(memv_write + 1, u02);

                ++memv_read;
                memv_write += 2;
                --lenv;
            }
        }

        mem_read = reinterpret_cast<TPACK*>(memv_read);
        mem_write = reinterpret_cast<T*>(memv_write);

        while (len-- > 0) {
            *(mem_write++) = VMT::template unshift_and_add<Shift>(*(mem_read++), offset);
        }

        assert(mem_read == reinterpret_cast<TPACK *>(mem_write));
        assert(mem_read == mem_end);
    }

    void align_vectorized(const T* left, const T* right,
                          const AH& hint, const TV P,
                          T*& read_left, T*& read_right,
                          T*& tmp_start_left, T*& tmp_left,
                          T*& tmp_start_right, T*& tmp_right) const
    {
        const auto left_align = hint.left_align;
        const auto right_align = hint.right_align;
        const auto rai = ~((right_align - 1) >> 31);
        const auto lai = left_align >> 31;
        const auto pre_aligned_left = (TV*)(left + left_align);
        const auto pre_aligned_right = (TV*)(right + right_align - N);

#ifndef NDEBUG
        memset((void *)_temp, 0, PARTITION_TMP_SIZE_IN_ELEMENTS * sizeof(T));
#endif

#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_vec_loads(2);
        vxsort_stats<T>::bump_vec_stores(4);
#endif
        /// :   left        :                 :     right
        /// :   │           :                 :         │
        /// :   │           :                 :         │
        /// :   ▼           :                 :         ▼
        /// ┌───┬───────────────────────────────────────┬─────┐
        /// │< <│< > > > > > x x x x x x x x x < < > > >│> > >│
        /// └┬──┴──────────┬───────────────────┬────────┴────┬┘
        /// :└──────┬──────┘:                 :└──────┬──────┘
        /// :└──┘       │   :                 :       │ └────┘
        /// :       ▼       :                 :       ▼
        ///    lt_vec                                 rt_vec
        // Alignment with vectorization is tricky, so read carefully before changing code:
        // 1. We load data, which we might need to align, if the alignment hints
        //    mean pre-alignment (or overlapping alignment)
        // 2. We partition and store in the following order:
        //    a) right-portion of right vector to the right-side
        //    b) left-portion of left vector to the left side
        //    c) at this point one-half of each partitioned vector has been committed
        //       back to memory.
        //    d) we advance the right write (tmp_right) pointer by how many elements
        //       were actually needed to be written to the right hand side
        //    e) We write the right portion of the left vector to the right side
        //       now that its write position has been updated
        auto rt_vec = VMT::load_vec(pre_aligned_right);
        auto lt_vec = VMT::load_vec(pre_aligned_left);
        const auto rt_mask = VMT::get_cmpgt_mask(rt_vec, P);
        const auto lt_mask = VMT::get_cmpgt_mask(lt_vec, P);
        const auto rt_popcount_right_part = std::max(_mm_popcnt_u32(rt_mask), right_align);
        const auto lt_popcount_right_part = _mm_popcnt_u32(lt_mask);
        const auto rt_popcount_left_part = N - rt_popcount_right_part;
        const auto lt_popcount_left_part = N - lt_popcount_right_part;

        // This is a compile-time check
        // controlled by the compiled instruction set
        if (VMT::supports_compress_writes()) {
            VMT::store_compress_vec((TV *) (tmp_right + N - rt_popcount_right_part), rt_vec, rt_mask);
            VMT::store_compress_vec((TV *)tmp_left, lt_vec, ~lt_mask);

          tmp_right -= rt_popcount_right_part & rai;
          read_right += (right_align - N) & rai;

          VMT::store_compress_vec((TV *) (tmp_right + N - lt_popcount_right_part), lt_vec, lt_mask);
          tmp_right -= lt_popcount_right_part & lai;
          tmp_left += lt_popcount_left_part & lai;
          tmp_start_left += -left_align & lai;
          read_left += (left_align + N) & lai;

          VMT::store_compress_vec((TV*)tmp_left, rt_vec, ~rt_mask);
          tmp_left += rt_popcount_left_part & rai;
          tmp_start_right -= right_align & rai;
        }
        else {
#ifdef VXSORT_STATS
            vxsort_stats<T>::bump_perms(2);
#endif
            rt_vec = VMT::partition_vector(rt_vec, rt_mask);
            lt_vec = VMT::partition_vector(lt_vec, lt_mask);
            VMT::store_vec((TV*)tmp_right, rt_vec);
            VMT::store_vec((TV*)tmp_left, lt_vec);

            tmp_right -= rt_popcount_right_part & rai;
            read_right += (right_align - N) & rai;

            VMT::store_vec((TV*)tmp_right, lt_vec);
            tmp_right -= lt_popcount_right_part & lai;

            tmp_left += lt_popcount_left_part & lai;
            tmp_start_left += -left_align & lai;
            read_left += (left_align + N) & lai;

            VMT::store_vec((TV*)tmp_left, rt_vec);
            tmp_left += rt_popcount_left_part & rai;
            tmp_start_right -= right_align & rai;
        }
    }

public:
    /**
     * Sort a given range
     * @param left The left edge of the range, including
     * @param right The right edge of the range, including
     * @param left_hint Optional; A hint, Use to speed up the sorting operation, describing a single value that is known to be
     *        smaller-than, or equal to all values contained within the provided array.
     * @param right_hint Optional; A hint, Use to speed up the sorting operation, describing a single value that is known to be
     *        larger-than than all values contained within the provided array.
     */
    NOINLINE void sort(T* left, T* right,
                       T left_hint = std::numeric_limits<T>::min(),
                       T right_hint = std::numeric_limits<T>::max())
    {
        init_isa_detection();

#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_sorts((right - left) + 1);
#endif
        reset(left, right);
        auto depth_limit = 2 * floor_log2_plus_one(right + 1 - left);
        sort(left, right, left_hint, right_hint, AH(), depth_limit);
    }
};

}  // namespace vxsort

#include "vxsort_targets_disable.h"

#endif
