#ifndef VXSORT_VXSORT_H
#define VXSORT_VXSORT_H

#include <cassert>

#include "alignment.h"
#include "defs.h"
#include "isa_detection.h"
#include "vector_machine/machine_traits.h"
#include "partition_machine.h"
#include "pack_machine.h"
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
    using PMT = partition_machine<T, M>;
    using PKM = pack_machine<T, M, Shift>;

    typedef typename VMT::TPACK TPACK;
    using VM_PACKED = vxsort_machine_traits<TPACK, M>;
    typedef typename VMT::TV TV;
    typedef alignment_hint<T, M> AH;

    static const i32 ELEMENT_ALIGN = sizeof(T) - 1;
    static const i32 N = sizeof(TV) / sizeof(T);
    static_assert(is_powerof2(N), "vector-size / element-size must be a power of 2");

    static const i32 SMALL_SORT_THRESHOLD_ELEMENTS = 1024;
    static const i32 SMALL_SORT_THRESHOLD_VECTORS = SMALL_SORT_THRESHOLD_ELEMENTS / N;
    static const i32 SLACK_PER_SIDE_IN_VECTORS = Unroll;
    static const size_t ALIGN = AH::ALIGN;
    static const size_t ALIGN_MASK = AH::ALIGN_MASK;
    static const i32 SLACK_PER_SIDE_IN_ELEMENTS = SLACK_PER_SIDE_IN_VECTORS * N;
    // The formula for figuring out how much temporary space we need for partitioning:
    // 2 x the number of slack elements on each side for the purpose of partitioning in unrolled manner +
    // 2 x amount of maximal bytes needed for alignment (32)
    // one more vector's worth of elements since we write with N-way stores from both ends of the temporary area
    // and we must make sure we do not accidentally over-write from left -> right or vice-versa right on that edge...
    // In other words, while we allocated this much temp memory, the actual amount of elements inside said memory
    // is smaller by 8 elements + 1 for each alignment (max alignment is actually N-1, I just round up to N...)
    // This long sense just means that we over-allocate N+2 elements...
    static const i32 PARTITION_SPILL_SIZE_IN_ELEMENTS =
            (2 * SLACK_PER_SIDE_IN_ELEMENTS + N + 4*N);

    static_assert(PARTITION_SPILL_SIZE_IN_ELEMENTS < SMALL_SORT_THRESHOLD_ELEMENTS, "Unroll-level must match small-sorting threshold");
    static const i32 PackUnroll = (Unroll / 2 > 0) ? Unroll / 2 : 1;


    void reset(T* start, T* end) {
        _depth = 0;
        _start = start;
        _end = end;
    }

    T* _start = nullptr;
    T* _end = nullptr;

    T _spill[PARTITION_SPILL_SIZE_IN_ELEMENTS];
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

    void sort(T* left, T* right,
              T left_hint, T right_hint,
              AH alignment, i32 depth_limit) {
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

        if (alignment.left_masked_amount == AH::REALIGN) {
            alignment.calc_left_alignment(left);
        }

        // `right` points to the last element of the array, where we the pivot is stored
        // IOW, while it *technically* points *inside* the partition, it is, actually
        // pointing exactly one element past the last element we will partition
        // Therefore, we pass it as-is since this is what the callee expects exactly
        if (alignment.right_unmasked_amount == AH::REALIGN) {
            alignment.calc_right_alignment(right);
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
        if (VMT::supports_packing()) {
            if (VMT::template can_pack<Shift>(right_hint - left_hint)) {
#ifdef VXSORT_STATS
                vxsort_stats<T>::bump_packs(length);
                vxsort_stats<T>::bump_unpacks(length);
#endif
                auto left_length = vectorized_packed_partition(left, right, left_hint, alignment);
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

        auto sep = vectorized_partition<Unroll>(left, right, alignment);

        _depth++;
        sort(left, sep - 2, left_hint, *sep, alignment.clear_right(), depth_limit);
        sort(sep, right, *(sep - 2), right_hint, alignment.clear_left(), depth_limit);
        _depth--;
    }

    template <typename TFriend, vector_machine MFriend, i32 UnrollFriend, i32 ShiftFriend>
    friend class vxsort;

    /// vectorized_partition - partition an array with vector operations
    /// \tparam InnerUnroll - The amount of unrolling / code-bloat
    /// \param left - pointer (inclusive) to the left edge of the partition
    /// \param right - pointer (inclusive) to the right edge of the partition.
    ///                Note: as part of the internal convention, this is a "dual" pointer
    ///                as it points to the pivot for this partition call, and is pointing just past
    ///                the last element to be partitioned during this call.
    /// \param alignment - a (partially) cache alignment used to communicate where the
    ///               the nearest vector-alignment left+right of the partition
    ///               is situated.
    /// \return A pointer to the new location
    ///
    /// Vectorized double-pumped (dual-sided) partitioning:
    /// We start with reading the pivot value passed on to use at the position
    /// pointed to by the 'right' parameter. We process the array from both ends.
    ///
    /// To get this rolling, we first read <InnerUnroll> x vector-sized elements
    /// from the left and right, each. This data is partitioned and stored in a
    /// small temporary buffer to make some room for the main block where an
    /// inplace partitioning loop is performed.
    /// Conceptually, the bulk
    /// of the processing looks like this after clearing out some initial space
    /// as described above:
    /// @code
    /// |-- InnerUnroll x -|                                    |-- InnerUnroll x -|
    ///        vector-size                                             vector-size
    ///          bytes                                                   bytes
    /// [..........................................................................]
    ///  ^wl                ^rl                                rr^                wr^
    /// Where:
    /// wl = write_left_v
    /// rl = read_left_v
    /// rr = read_right_v
    /// wr = write_right_v
    /// @endcode
    ///
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
    T* vectorized_partition(T* const left, T* const right, const AH alignment)
    {
//#ifdef NDEBUG
//#   define LOAD_LEFT
//#   define LOAD_RIGHT
//#else
//        usize last_left = (usize)left;
//        usize last_right = (usize)right;
//        #define LOAD_LEFT(p) { fmt::print("LL: {}\n", (usize) p - last_left); last_left = (usize) p; }
//        #define LOAD_RIGHT(p) { fmt::print("LR: {}\n", last_right - (usize) p); last_right = (usize) p; }
//#endif

        assert(right - left >= SMALL_SORT_THRESHOLD_ELEMENTS);
        assert((reinterpret_cast<usize>(left) & ELEMENT_ALIGN) == 0);
        assert((reinterpret_cast<usize>(right) & ELEMENT_ALIGN) == 0);
#ifndef NDEBUG
        memset((void *)_spill, 0, PARTITION_SPILL_SIZE_IN_ELEMENTS * sizeof(T));
#endif


#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_partitions((right - left) + 1);
#endif

        // An additional unfortunate complexity we need to deal with is that the
        // right pointer must be decremented by another Vector256<T>.Count elements
        // Since the Load/Store primitives obviously accept start addresses
        auto pivot = *right;

        // Broadcast the selected pivot
        const auto P = VMT::broadcast(pivot);

        auto * RESTRICT spill_read_left = _spill;
        auto * RESTRICT spill_write_left = spill_read_left;
        auto * RESTRICT spill_read_right = _spill + PARTITION_SPILL_SIZE_IN_ELEMENTS;
        auto * RESTRICT spill_write_right = spill_read_right;

        // mutable pointer copies of the originals
        auto * RESTRICT read_left = left;
        auto * RESTRICT read_right = right;

        // the read heads always advance by N elements towards te middle,
        // It would be wise to spend some extra effort here to align the read
        // pointers such that they align naturally to vector size;
        // on vector-machines, where the ratio between vector/cache-line size
        // is close, for example, assuming 64-byte cache-line:
        // * unaligned 256-bit loads create split-line loads 50% of the time
        // * unaligned 512-bit loads create a split-line loads 100% of the time
        PMT::align_vectorized(alignment.left_masked_amount,
                              alignment.right_unmasked_amount,
                              P,
                              read_left, read_right,
                              spill_read_left, spill_write_left,
                              spill_read_right, spill_write_right);

        assert((right - left) ==
               ((read_right + N) - read_left) + // Unpartitioned elements (+N for right-side vec reads)
               (spill_write_left - spill_read_left) + // partitioned to left-spill
               (spill_read_right - (spill_write_right + N))); // partitioned to right-spill (+N for right-side vec reads)



        assert(((usize)read_left & ALIGN_MASK) == 0);
        assert(((usize)read_right & ALIGN_MASK) == 0);
        assert((((usize)read_right - (usize)read_left) % ALIGN) == 0);
        assert((read_right - read_left) >= InnerUnroll * 2);

        // From now on, we are fully aligned
        // and all reading is done in full vector units
        auto * RESTRICT read_left_v = reinterpret_cast<TV*>(read_left);
        auto * RESTRICT read_right_v = reinterpret_cast<TV*>(read_right);

#ifndef NDEBUG
        read_left = nullptr;
        read_right = nullptr;
#endif

        for (auto u = 0; u < InnerUnroll; u++) {
            auto dl = VMT::load_vec(read_left_v + u);
            auto dr = VMT::load_vec(read_right_v - u);
            PMT::partition_block(dl, P, spill_write_left, spill_write_right);
            PMT::partition_block(dr, P, spill_write_left, spill_write_right);
        }

        // Fix-up spill_write_right after the last vector operation
        // potentially *writing* through it is done
        spill_write_right += N;
        // Adjust for the reading that was made above
        read_left_v += InnerUnroll;
        read_right_v += 1;
        read_right_v -= InnerUnroll*2;
        TV* nextPtr;

        auto * RESTRICT write_left = left;
        auto * RESTRICT write_right = right - N;

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
                case 12: PMT::partition_block(d12, P, write_left, write_right); [[fallthrough]];
                case 11: PMT::partition_block(d11, P, write_left, write_right); [[fallthrough]];
                case 10: PMT::partition_block(d10, P, write_left, write_right); [[fallthrough]];
                case  9: PMT::partition_block(d09, P, write_left, write_right); [[fallthrough]];
                case  8: PMT::partition_block(d08, P, write_left, write_right); [[fallthrough]];
                case  7: PMT::partition_block(d07, P, write_left, write_right); [[fallthrough]];
                case  6: PMT::partition_block(d06, P, write_left, write_right); [[fallthrough]];
                case  5: PMT::partition_block(d05, P, write_left, write_right); [[fallthrough]];
                case  4: PMT::partition_block(d04, P, write_left, write_right); [[fallthrough]];
                case  3: PMT::partition_block(d03, P, write_left, write_right); [[fallthrough]];
                case  2: PMT::partition_block(d02, P, write_left, write_right); [[fallthrough]];
                case  1: PMT::partition_block(d01, P, write_left, write_right);
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
            PMT::partition_block(d, P, write_left, write_right);
        }

        // 3. Copy-back the 4 registers + remainder we partitioned in the beginning
        auto left_tmp_size = spill_write_left - spill_read_left;
        memcpy(write_left, spill_read_left, left_tmp_size * sizeof(T));
        write_left += left_tmp_size;
        auto right_tmp_size = spill_read_right - spill_write_right;
        memcpy(write_left, spill_write_right, right_tmp_size * sizeof(T));

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
    /// \param alignment - a (partially) cache alignment used to communicate where the
    ///               the nearest vector-alignment left+right of the partition
    ///               is situated.
    /// \return The amount of elements partitioned to the left side
    size_t vectorized_packed_partition(T* const left, T* const right,
                                       T min_bounding, const AH alignment) {
        assert(right - left >= SMALL_SORT_THRESHOLD_ELEMENTS);
        assert((reinterpret_cast<size_t>(left) & ELEMENT_ALIGN) == 0);
        assert((reinterpret_cast<size_t>(right) & ELEMENT_ALIGN) == 0);

#ifndef NDEBUG
        memset((void *)_spill, 0, PARTITION_SPILL_SIZE_IN_ELEMENTS * sizeof(T));
#endif


#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_partitions((right - left) + 1);
#endif

        auto pivot = *right;

        // Broadcast the selected pivot
        const auto P = VMT::broadcast(pivot);
        // Create a vectorized version of the offset by which we need to
        // correct the data before packing it
        auto constexpr MIN = T(std::numeric_limits<TPACK>::min());
        auto offset = VMT::template shift_n_sub<Shift>(min_bounding, MIN);
        const TV offset_v = VMT::broadcast(offset);

        auto * RESTRICT read_left = left;
        auto * RESTRICT read_right = right;

        auto * RESTRICT spill_read_left = _spill;
        auto * RESTRICT spill_write_left = spill_read_left;
        auto * RESTRICT spill_read_right = _spill + PARTITION_SPILL_SIZE_IN_ELEMENTS;
        auto * RESTRICT spill_write_right = spill_read_right;

        // the read heads always advance by N elements towards te middle,
        // It would be wise to spend some extra effort here to align the read
        // pointers such that they align naturally to vector size;
        // on vector-machines, where the ratio between vector/cache-line size
        // is close, for example, assuming 64-byte cache-line:
        // * unaligned 256-bit loads create split-line loads 50% of the time
        // * unaligned 512-bit loads create a split-line loads 100% of the time
        PMT::align_vectorized(alignment.left_masked_amount,
                              alignment.right_unmasked_amount,
                              P,
                              read_left, read_right,
                              spill_read_left, spill_write_left,
                              spill_read_right, spill_write_right);

        assert(((size_t)read_left & ALIGN_MASK) == 0);
        assert(((size_t)read_right & ALIGN_MASK) == 0);

        assert((((size_t)read_right - (size_t)read_left) % ALIGN) == 0);
        //assert((readRight - readLeft) >= InnerUnroll * 2);

        // From now on, we are fully aligned
        // and all reading is done in full vector units
        auto * RESTRICT read_left_v = reinterpret_cast<TV*>(read_left);
        auto * RESTRICT read_right_v = reinterpret_cast<TV*>(read_right);
#ifndef NDEBUG
        read_left = nullptr;
        read_right = nullptr;
#endif

        auto* RESTRICT write_left = reinterpret_cast<TPACK*>(left);
        auto* RESTRICT write_right = reinterpret_cast<TPACK*>(right+1) - 2*N;

        // We will be packing before partitioning, so
        // We must generate a pre-packed pivot
        const auto packed_pivot = VMT::template shift_n_sub<Shift>(pivot, offset);
        const TV PPP = VM_PACKED::broadcast(static_cast<TPACK>(packed_pivot));

        auto len_v = read_right_v - read_left_v + 1;
        auto len_dv = len_v / 2;

        len_v -= len_dv * 2;

#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_vec_loads(2 * len_dv);
        vxsort_stats<T>::bump_vec_stores(2 * len_dv);
#endif

        for (auto i = 0; i < len_dv; i++) {
            auto dl = VMT::load_vec(read_left_v + i);
            auto dr = VMT::load_vec(read_right_v - i);

            auto packed_data  = PKM::pack_vectors(dl, dr, offset_v);

            vxsort<TPACK, M, Unroll>::PMT::partition_block(packed_data, PPP, write_left, write_right);
        }

        // We might have one more vector worth of stuff to partition, so we'll do it with
        // scalar partitioning into the tmp space
        if (len_v > 0) {
            auto slack = VMT::load_vec((TV *) (read_left_v + len_dv));
            PMT::partition_block(slack, P, spill_write_left, spill_write_right);
        }
        // Fix-up spill_write_right after the last vector operation
        // potentially *writing* through it is done
        spill_write_right += N;

        write_right += 2*N;

        for (auto *p = spill_read_left; p < spill_write_left; p++) {
            *(write_left++) = static_cast<TPACK>(VMT::template shift_n_sub<Shift>(*p, offset));
        }

        for (auto *p = spill_write_right; p < spill_read_right; p++) {
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
    /// @code
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
    /// @endcode


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
                PKM::unpack_vectors(base_v, d01, u01, u02);
                VMT::store_vec(memv_write + 0, u01);
                VMT::store_vec(memv_write + 1, u02);
                if (UnpackUnroll == 1) break;
                PKM::unpack_vectors(base_v, d02, u03, u04);
                VMT::store_vec(memv_write - 2, u03);
                VMT::store_vec(memv_write - 1, u04);
                if (UnpackUnroll == 2) break;
                PKM::unpack_vectors(base_v, d03, u05, u06);
                VMT::store_vec(memv_write - 4, u05);
                VMT::store_vec(memv_write - 3, u06);
                if (UnpackUnroll == 3) break;
                PKM::unpack_vectors(base_v, d04, u07, u08);
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

                PKM::unpack_vectors(base_v, d01, u01, u02);
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
                PKM::unpack_vectors(base_v, d01, u01, u02);
                VMT::store_vec(memv_write + 0, u01);
                VMT::store_vec(memv_write + 1, u02);
                if (UnpackUnroll == 1) break;
                PKM::unpack_vectors(base_v, d02, u03, u04);
                VMT::store_vec(memv_write + 2, u03);
                VMT::store_vec(memv_write + 3, u04);
                if (UnpackUnroll == 2) break;
                PKM::unpack_vectors(base_v, d03, u05, u06);
                VMT::store_vec(memv_write + 4, u05);
                VMT::store_vec(memv_write + 5, u06);
                if (UnpackUnroll == 3) break;
                PKM::unpack_vectors(base_v, d04, u07, u08);
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

                PKM::unpack_vectors(base_v, d01, u01, u02);
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

#endif
