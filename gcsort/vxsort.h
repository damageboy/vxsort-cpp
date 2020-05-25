#ifndef GCSORT_VXSORT_H
#define GCSORT_VXSORT_H

#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

#include <smallsort/bitonic_sort.int64_t.generated.h>
#include <algorithm>
#include <cstdio>
#include <cstring>

namespace gcsort {
using gcsort::smallsort::bitonic;

template <typename T>
class vxsort {
 public:
  static void sort(T* ptr, int length);
};

template <>
class vxsort<int64_t> {
 private:
  static const int32_t SLACK_PER_SIDE_IN_VECTORS = 1;
  static const int32_t SMALL_SORT_THRESHOLD_ELEMENTS = 64;
  static const size_t ALIGN = 32;
  static const size_t ALIGN_MASK = ALIGN - 1;

  static const int N = 4;
  static const int SLACK_PER_SIDE_IN_ELEMENTS = SLACK_PER_SIDE_IN_VECTORS * N;
  // We allocate the amount of slack space + up-to 2 more alignment blocks
  static const int PARTITION_TMP_SIZE_IN_ELEMENTS =
      (2 * SLACK_PER_SIDE_IN_ELEMENTS + N + ((2 * ALIGN) / sizeof(long)));

  int64_t* _startPtr;
  int64_t* _endPtr;

  int64_t _temp[PARTITION_TMP_SIZE_IN_ELEMENTS];
  int _depth;

  static int floor_log2_plus_one(int n) {
    auto result = 0;
    while (n >= 1) {
      result++;
      n /= 2;
    }
    return result;
  }

  template <typename T>
  static void swap(T* left, T* right) {
    auto tmp = *left;
    *left = *right;
    *right = tmp;
  }

  template <typename T>
  static void swap_if_greater(T* left, T* right) {
    if (*left <= *right)
      return;
    swap(left, right);
  }

  template <typename T>
  static void insertion_sort(T* lo, T* hi) {
    for (auto i = lo + 1; i <= hi; i++) {
      auto j = i;
      auto t = *i;
      while ((j > lo) && (t < *(j - 1))) {
        *j = *(j - 1);
        j--;
      }
      *j = t;
    }
  }

  template <typename T>
  static void heap_sort(T* lo, T* hi) {
    size_t n = hi - lo + 1;
    for (size_t i = n / 2; i >= 1; i--) {
      down_heap(i, n, lo);
    }
    for (size_t i = n; i > 1; i--) {
      swap(lo, lo + i - 1);
      down_heap(1, i - 1, lo);
    }
  }

  template <typename T>
  static void down_heap(size_t i, size_t n, T* lo) {
    auto d = *(lo + i - 1);
    size_t child;
    while (i <= n / 2) {
      child = 2 * i;
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

  struct alignment_hint {
   public:
    static const int8_t REALIGN = 0x66;

    alignment_hint() : left_align(REALIGN), right_align(REALIGN) {}
    alignment_hint realign_left() {
      alignment_hint copy = *this;
      copy.left_align = REALIGN;
      return copy;
    }

    alignment_hint realign_right() {
      alignment_hint copy = *this;
      copy.left_align = REALIGN;
      return copy;
    }

    int left_align : 8;
    int right_align : 8;
  };

  void reset(int64_t* start, int64_t* end) {
    _depth = 0;
    _startPtr = start;
    _endPtr = end;
  }

  bool is_aligned(void *p) {
    return (size_t)p % ALIGN == 0;
  }

  void sort(int64_t* left,
            int64_t* right,
            alignment_hint realignHint,
            int depthLimit) {
    auto length = (size_t)(right - left + 1);

    int64_t* mid;
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

      auto nextLength = (length & 7) > 0 ? (length + 8) & ~7 : length;

      auto extraSpaceNeeded = nextLength - length;
      auto fakeLeft = left - extraSpaceNeeded;
      if (fakeLeft >= _startPtr) {
        bitonic<int64_t>::sort(fakeLeft, nextLength);
      } else {
        insertion_sort(left, right);
      }
      return;
    }

    // Detect a whole bunch of bad cases where partitioning
    // will not do well:
    // 1. Reverse sorted array
    // 2. High degree of repeated values (dutch flag problem, one value)
    if (depthLimit == 0) {
      heap_sort(left, right);
      _depth--;
      return;
    }
    depthLimit--;

    // This is going to be a bit weird:
    // Pre/Post alignment calculations happen here: we prepare hints to the
    // partition function of how much to align and in which direction
    // (pre/post). The motivation to do these calculations here and the actual
    // alignment inside the partitioning code is that here, we can cache those
    // calculations. As we recurse to the left we can reuse the left cached
    // calculation, And when we recurse to the right we reuse the right
    // calculation, so we can avoid re-calculating the same aligned addresses
    // throughout the recursion, at the cost of a minor code complexity
    // Since we branch on the magi values REALIGN_LEFT & REALIGN_RIGHT its safe
    // to assume the we are not torturing the branch predictor.'

    // We use a long as a "struct" to pass on alignment hints to the
    // partitioning By packing 2 32 bit elements into it, as the JIT seem to not
    // do this. In reality  we need more like 2x 4bits for each side, but I
    // don't think there is a real difference'

    if (realignHint.left_align == alignment_hint::REALIGN) {
      // Alignment flow:
      // * Calculate pre-alignment on the left
      // * See it would cause us an out-of bounds read
      // * Since we'd like to avoid that, we adjust for post-alignment
      // * There are no branches since we do branch->arithmetic
      auto preAlignedLeft = (int64_t*) ((size_t)left & ~ALIGN_MASK);
      auto cannotPreAlignLeft = (preAlignedLeft - _startPtr) >> 63;
      realignHint.left_align = (preAlignedLeft - left) + (N & cannotPreAlignLeft);
      assert(is_aligned(left + realignHint.left_align) && (left + realignHint.left_align));
    }

    if (realignHint.right_align == alignment_hint::REALIGN) {
      // right is pointing just PAST the last element we intend to partition
      // (where we also store the pivot) So we calculate alignment based on
      // right - 1, and YES: I am casting to ulong before doing the -1, this is
      // intentional since the whole thing is either aligned to 32 bytes or not,
      // so decrementing the POINTER value by 1 is sufficient for the alignment,
      // an the JIT sucks at this anyway
      auto preAlignedRight =
          (int64_t*) (((size_t)(right - 1) & ~ALIGN_MASK) + ALIGN);
      auto cannotPreAlignRight = (_endPtr - preAlignedRight) >> 63;
      realignHint.right_align = (preAlignedRight - right - (N & cannotPreAlignRight));
      assert(is_aligned(right + realignHint.right_align) && right + realignHint.right_align);
    }


    // Compute median-of-three, of:
    // the first, mid and one before last elements
    mid = left + ((right - left) / 2);
    swap_if_greater(left, mid);
    swap_if_greater(left, right - 1);
    swap_if_greater(mid, right - 1);

    // Pivot is mid, place it in the right hand side
    swap(mid, right);

    auto sep = vectorized_partition(left, right, realignHint);

    _depth++;
    sort(left, sep - 2, realignHint.realign_right(), depthLimit);
    sort(sep, right, realignHint.realign_left(), depthLimit);
    _depth--;
  }

  const static int8_t perm_table[128];

  __m256i get_perm(int mask) {
    assert(mask >= 0);
    assert(mask <= 15);
    return _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(perm_table + mask * 8)));
  }

  __attribute__((noinline))
  int64_t* align_left_scalar_uncommon(int64_t* read_left,
                                   int64_t pivot,
                                   int64_t*& tmp_left,
                                   int64_t*& tmp_right) {
    if (((uint64_t)read_left & ALIGN_MASK) == 0)
      return read_left;

    auto next_align = (int64_t*)(((uint64_t)read_left + ALIGN) & ~ALIGN_MASK);
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

  __attribute__((noinline))
  int64_t* align_right_scalar_uncommon(int64_t* readRight,
                                    int64_t pivot,
                                    int64_t*& tmpLeft,
                                    int64_t*& tmpRight) {
    if (((uint64_t)readRight & ALIGN_MASK) == 0)
      return readRight;

    auto nextAlign = (int64_t*)((uint64_t)readRight & ~ALIGN_MASK);
    while (readRight > nextAlign) {
      auto v = *(--readRight);
      if (v <= pivot) {
        *(tmpLeft++) = v;
      } else {
        *(--tmpRight) = v;
      }
    }

    return readRight;
  }

  int64_t* vectorized_partition(int64_t* left, int64_t* right, alignment_hint hint) {
    assert(right - left >= SMALL_SORT_THRESHOLD_ELEMENTS);
    assert(((size_t)left & 0x7) == 0);
    assert(((size_t)right & 0x7) == 0);

    // Vectorized double-pumped (dual-sided) partitioning:
    // We start with picking a pivot using the media-of-3 "method"
    // Once we have sensible pivot stored as the last element of the array
    // We process the array from both ends.
    //
    // To get this rolling, we first read 2 Vector256 elements from the left and
    // another 2 from the right, and store them in some temporary space in order
    // to leave enough "space" inside the vector for storing partitioned values.
    // Why 2 from each side? Because we need n+1 from each side where n is the
    // number of Vector256 elements we process in each iteration... The
    // reasoning behind the +1 is because of the way we decide from *which* side
    // to read, we may end up reading up to one more vector from any given side
    // and writing it in its entirety to the opposite side (this becomes
    // slightly clearer when reading the code below...) Conceptually, the bulk
    // of the processing looks like this after clearing out some initial space
    // as described above:

    // [.............................................................................]
    //  ^wl          ^rl                                               rr^ wr^
    // Where:
    // wl = writeLeft
    // rl = readLeft
    // rr = readRight
    // wr = writeRight

    // In every iteration, we select what side to read from based on how much
    // space is left between head read/write pointer on each side...
    // We read from where there is a smaller gap, e.g. that side
    // that is closer to the unfortunate possibility of its write head
    // overwriting its read head... By reading from THAT side, we're ensuring
    // this does not happen

    // An additional unfortunate complexity we need to deal with is that the
    // right pointer must be decremented by another Vector256<T>.Count elements
    // Since the Load/Store primitives obviously accept start addresses
    auto pivot = *right;
    // We do this here just in case we need to pre-align to the right
    // We end up
    *right = INT64_MAX;

    const __m256i P = _mm256_set1_epi64x(pivot);

    auto readLeft = left;
    auto readRight = right;
    auto writeLeft = left;
    auto writeRight = right - N;

    auto tmpStartLeft = _temp;
    auto tmpLeft = tmpStartLeft;
    auto tmpStartRight = _temp + PARTITION_TMP_SIZE_IN_ELEMENTS;
    auto tmpRight = tmpStartRight;

    // Broadcast the selected pivot
    tmpRight -= N;

    // the read heads always advance by 8 elements, or 32 bytes,
    // We can spend some extra time here to align the pointers
    // so they start at a cache-line boundary
    // Once that happens, we can read with Avx.LoadAlignedVector256
    // And also know for sure that our reads will never cross cache-lines
    // Otherwise, 50% of our AVX2 Loads will need to read from two cache-lines

    auto leftAlign = hint.left_align;
    auto rightAlign = hint.right_align;

    auto preAlignedLeft = left + leftAlign;
    auto preAlignedRight = right + rightAlign - N;

    // Read overlapped data from right (includes re-reading the pivot)
    auto RT0 = _mm256_lddqu_si256((__m256i*)preAlignedRight);
    auto LT0 = _mm256_lddqu_si256((__m256i*)preAlignedLeft);
    auto rtMask = _mm256_movemask_pd(_mm256_cmpgt_epi64(RT0, P));
    auto ltMask = _mm256_movemask_pd(_mm256_cmpgt_epi64(LT0, P));
    auto rtPopCount = std::max(_mm_popcnt_u32(rtMask), rightAlign);
    auto ltPopCount = _mm_popcnt_u32(ltMask);
    RT0 = _mm256_permutevar8x32_epi32(RT0, get_perm(rtMask));
    LT0 = _mm256_permutevar8x32_epi32(LT0, get_perm(ltMask));
    _mm256_storeu_si256((__m256i *) tmpRight, RT0);
    _mm256_storeu_si256((__m256i *) tmpLeft, LT0);

    auto rai = ~((rightAlign - 1) >> 31);
    auto lai = leftAlign >> 31;

    tmpRight -= rtPopCount & rai;
    rtPopCount = 4 - rtPopCount;
    readRight += (rightAlign - N) & rai;

    _mm256_storeu_si256((__m256i *) tmpRight, LT0);
    tmpRight -= ltPopCount & lai;
    ltPopCount = 4 - ltPopCount;
    tmpLeft += ltPopCount & lai;
    tmpStartLeft += -leftAlign & lai;
    readLeft += (leftAlign + N) & lai;

    _mm256_storeu_si256((__m256i *) tmpLeft, RT0);
    tmpLeft += rtPopCount & rai;
    tmpStartRight -= rightAlign & rai;

    if (leftAlign > 0) {
      tmpRight += N;
      readLeft = align_left_scalar_uncommon(readLeft, pivot, tmpLeft, tmpRight);
      tmpRight -= N;
    }

    if (rightAlign < 0) {
      tmpRight += N;
      readRight =
          align_right_scalar_uncommon(readRight, pivot, tmpLeft, tmpRight);
      tmpRight -= N;
    }
    assert(((size_t)readLeft & ALIGN_MASK) == 0);
    assert(((size_t)readRight & ALIGN_MASK) == 0);

    assert((((uint8_t*)readRight - (uint8_t*)readLeft) % ALIGN) == 0);
    assert((readRight - readLeft) >= SLACK_PER_SIDE_IN_ELEMENTS * 2);

    partition_block(readLeft  + 0 * N, P, tmpLeft, tmpRight);
    partition_block(readRight - 1 * N, P, tmpLeft, tmpRight);

    tmpRight += N;
    // Adjust for the reading that was made above
    readLeft  += 1 * N;
    readRight -= 2 * N;

    while (readRight >= readLeft) {
      int64_t* nextPtr;
      if (writeRight - readRight < N) {
        nextPtr = readRight;
        readRight -= N;
      } else {
        nextPtr = readLeft;
        readLeft += N;
      }

      partition_block(nextPtr, P, writeLeft, writeRight);
    }

    // 3. Copy-back the 4 registers + remainder we partitioned in the beginning
    auto leftTmpSize = tmpLeft - tmpStartLeft;
    memcpy(writeLeft, tmpStartLeft, leftTmpSize * sizeof(int64_t));
    writeLeft += leftTmpSize;
    auto rightTmpSize = tmpStartRight - tmpRight;
    memcpy(writeLeft, tmpRight, rightTmpSize * sizeof(int64_t));

    // Shove to pivot back to the boundary
    *right = *writeLeft;
    *writeLeft++ = pivot;

    assert(writeLeft > left);
    assert(writeLeft <= right);

    return writeLeft;
  }

  void partition_block(int64_t* dataPtr,
                       const __m256i& P,
                       int64_t*& writeLeft,
                       int64_t*& writeRight) {
    auto dataVec = _mm256_load_si256((__m256i*)dataPtr);
    auto mask = _mm256_movemask_pd(_mm256_cmpgt_epi64(dataVec, P));
    dataVec = _mm256_permutevar8x32_epi32(dataVec, get_perm(mask));
    _mm256_storeu_si256((__m256i*)writeLeft, dataVec);
    _mm256_storeu_si256((__m256i*)writeRight, dataVec);
    auto popCount = -_mm_popcnt_u64(mask);
    writeRight += popCount;
    writeLeft += popCount + N;
  }

 public:
  void sort(int64_t* left, int64_t* right) {
    reset(left, right);
    auto depthLimit = 2 * floor_log2_plus_one(right + 1 - left);
    sort(left, right, alignment_hint(), depthLimit);
  }

};
}  // namespace gcsort
#endif
