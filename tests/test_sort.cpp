#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "util.h"

#include <introsort.h>
#include <vxsort.h>
#include <smallsort/bitonic_sort.h>

#include <algorithm>
#include <iterator>
#include <random>

namespace gcsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;



// A new one of these is create for each test
struct SortTest : public testing::TestWithParam<int> {
 protected:
  std::vector<int64_t> V;

 public:
  virtual void SetUp() {
    V = std::vector<int64_t>(GetParam());
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

struct SmallSortTest : public SortTest {};

INSTANTIATE_TEST_SUITE_P(BitonicSizes,
                         SmallSortTest,
                         ValuesIn(range(4, 64, 4)),
                         PrintValue());

TEST_P(SmallSortTest, BitonicSort) {
  auto begin = V.data();

  gcsort::smallsort::bitonic<int64_t>::sort((int64_t*)begin, GetParam());

  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

struct FullSortTest : public SortTest {};

INSTANTIATE_TEST_SUITE_P(IntroSortSizes,
                         FullSortTest,
                         ValuesIn(multiply_range(10, 1000000, 10)),
                         PrintValue());


TEST_P(FullSortTest, IntroSort) {
  auto begin = (uint8_t **)V.data();
  auto end = (uint8_t **)V.data() + V.size() - 1;

  sort_introsort(begin, end);
  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

TEST_P(FullSortTest, VxSort) {
  auto begin = V.data();
  auto end = V.data() + V.size() - 1;

  auto sorter = gcsort::vxsort<int64_t>();
  sorter.sort(begin, end);

  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

}  // namespace gcsort_tests