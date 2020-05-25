#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "util.h"

#include <introsort.h>
#include <vxsort.h>
#include <smallsort/bitonic_sort.h>
#include <smallsort/bitonic_sort.int32_t.generated.h>
#include <smallsort/bitonic_sort.int64_t.generated.h>

#include <algorithm>
#include <iterator>
#include <random>

namespace gcsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;



// A new one of these is create for each test
template<typename T>
struct SortTest : public testing::TestWithParam<int> {
 protected:
  std::vector<T> V;

 public:
  virtual void SetUp() {
    V = std::vector<T>(GetParam());
    generate_unique_ptrs_vec(V);
  }
  virtual void TearDown() {}
};

struct PrintValue {
  template <class ParamType>
  std::string operator()(const testing::TestParamInfo<ParamType>& info) const {
    auto v = static_cast<int>(info.param);
    return std::to_string(v);
  }
};

struct SmallSortTest_i32 : public SortTest<int32_t> {};
struct SmallSortTest_i64 : public SortTest<int64_t> {};
struct SmallSortTest_float : public SortTest<float> {};
struct SmallSortTest_double : public SortTest<double> {};

INSTANTIATE_TEST_SUITE_P(BitonicSizes,
                         SmallSortTest_i64,
                         ValuesIn(range(4, 64, 4)),
                         PrintValue());

TEST_P(SmallSortTest_i64, BitonicSort) {
  auto begin = V.data();
  gcsort::smallsort::bitonic<int64_t>::sort((int64_t*)begin, GetParam());
  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

INSTANTIATE_TEST_SUITE_P(BitonicSizes,
                         SmallSortTest_i32,
                         ValuesIn(range(8, 128, 8)),
                         PrintValue());

TEST_P(SmallSortTest_i32, BitonicSort) {
    auto begin = V.data();
    gcsort::smallsort::bitonic<int32_t>::sort((int32_t*)begin, GetParam());
    EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}


struct FullSortTest_i32 : public SortTest<int32_t> {};
struct FullSortTest_i64 : public SortTest<int64_t> {};
struct FullSortTest_float : public SortTest<float> {};
struct FullSortTest_double : public SortTest<double> {};

INSTANTIATE_TEST_SUITE_P(IntroSortSizes,
                         FullSortTest_i64,
                         ValuesIn(multiply_range(10, 1000000, 10)),
                         PrintValue());


TEST_P(FullSortTest_i64, IntroSort) {
      auto begin = (uint8_t **)V.data();
      auto end = (uint8_t **)V.data() + V.size() - 1;

      sort_introsort(begin, end);
      EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

TEST_P(FullSortTest_i64, VxSort) {
      auto begin = V.data();
      auto end = V.data() + V.size() - 1;

      auto sorter = gcsort::vxsort<int64_t>();
      sorter.sort(begin, end);

      EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

INSTANTIATE_TEST_SUITE_P(IntroSortSizes,
                         FullSortTest_i32,
                         ValuesIn(multiply_range(10, 1000000, 10)),
                         PrintValue());


TEST_P(FullSortTest_i32, VxSort) {
    auto begin = V.data();
    auto end = V.data() + V.size() - 1;

    auto sorter = gcsort::vxsort<int32_t, 8>();
    sorter.sort(begin, end);

    EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}


}  // namespace gcsort_tests