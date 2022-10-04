#ifndef VXSORT_ALIGNNMENT_H
#define VXSORT_ALIGNNMENT_H

#include <cstdint>
#include "vector_machine/machine_traits.h"
#include "defs.h"

namespace vxsort {
using namespace vxsort::types;

using namespace std;

/// Perform vector sized alignment of array boundary reads (beginning/end)
/// \tparam T the primitive type being aligned
/// \tparam M the vector_machine being use (e.g. determines the vector width in bytes)
template <typename T, vector_machine M>
struct alignment_hint {
    using VMT = vxsort_machine_traits<T, M>;
    static constexpr i32 N = VMT::N;
    static constexpr usize ALIGN = sizeof(typename VMT::TV);
public:
    static const size_t ALIGN_MASK = ALIGN - 1;
    static const i8 REALIGN = 0x66;
    static_assert(REALIGN > ALIGN, "REALIGN must be larger than ALIGN");

    alignment_hint() : left_masked_amount(REALIGN), right_unmasked_amount(REALIGN) {}
    alignment_hint clear_left() {
        alignment_hint copy = *this;
        copy.left_masked_amount = REALIGN;
        return copy;
    }

    alignment_hint clear_right() {
        alignment_hint copy = *this;
        copy.right_unmasked_amount = REALIGN;
        return copy;
    }

    static bool is_aligned(const void* p) { return (usize)p % ALIGN == 0; }

    /// Perform "left-side"/beginning-of partition alignment.
    /// Given an inclusive pointer to the left-most element/beginning-of an array,
    /// alignment to the nearest whole vector sized pointer is performed, updating the
    /// internal `left_masked_amount` member with the number of elements to be masked
    /// off during a vector read.
    /// @param[in] p a pointer to the first element that is desired to be read
    void calc_left_alignment(const T *p) {
        // Alignment flow:
        // * Calculate pre-alignment position on the left
        // * convert to a valid input to be use with `generate_suffix_mask`
        const auto* pre_aligned_left = reinterpret_cast<T*>(reinterpret_cast<usize>(p) & ~ALIGN_MASK);
        left_masked_amount = p - pre_aligned_left;
        assert(left_masked_amount >= 0 && left_masked_amount < N);
        assert(is_aligned(pre_aligned_left));
    }

    /// Perform "right-side"/end-of an partition alignment. Given an exclusive pointer just past
    /// the right-most/end-of an array, alignment to the nearest vector sized read
    /// is performed, updating the internal `right_unmasked_amount` with the number of unmasked
    /// elements to be read
    /// @param[in] p a pointer past the last element that is desired to be read
    void calc_right_alignment(const T *p) {
        //    │01234567│01234567
        // (1)│xxxxxxxx│xxxp••••
        // (2)│xxxxxxxx│xp••••••
        // (3)│xxxxxxxx│p••••••
        //    p -> the parameter
        //    x -> data to be read
        //    • -> Masked elements
        // right_unmasked_amount should be:
        // (1) -> 3
        // (2) -> 1
        // (3) -> 8 (8/0 are the same in terms of masking)
        const auto* pre_aligned_right = reinterpret_cast<T*>(reinterpret_cast<usize>(p-1) & ~ALIGN_MASK);
        right_unmasked_amount = p - pre_aligned_right;
        assert(right_unmasked_amount >= 0 && right_unmasked_amount <= N);
        assert(is_aligned(pre_aligned_right));
    }

    i32 left_masked_amount : 8;
    i32 right_unmasked_amount : 8;
};

} // namespace vxsort
#endif  // VXSORT_ALIGNNMENT_H
