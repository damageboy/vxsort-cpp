 #include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <gcsort.h>
#include <bitonic_sort.h>

#include <random>
#include <algorithm>
#include <iterator>

namespace gcsort_tests {
    using testing::WhenSorted;
    using testing::ElementsAreArray;


    static std::vector<uint8_t *> generate_unique_ptrs_vec(size_t n) {
        std::vector<uint8_t *> pvec(n);

        std::iota(pvec.begin(), pvec.end(), (uint8_t *) 0x1000);

        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(pvec.begin(), pvec.end(), g);
        return pvec;
    }

    TEST(Sort, IntroSort) {
        auto v = generate_unique_ptrs_vec(1000);

        auto begin = v.data();
        auto end = v.data() + v.size() - 1;

        sort_introsort(begin, end);
        EXPECT_THAT(v, WhenSorted(ElementsAreArray(v)));
    }

    // A new one of these is create for each test
    class SortTest : public testing::TestWithParam<int>
    {
    public:
      virtual void SetUp(){}
      virtual void TearDown(){}
    };

  struct PrintValue {
    template <class ParamType>
    std::string operator()( const testing::TestParamInfo<ParamType>& info ) const
    {
      auto v = static_cast<int>(info.param);
      return std::to_string(v);
    }
  };

    INSTANTIATE_TEST_SUITE_P(
        BitonicSizes,
        SortTest,
        ::testing::Values(4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64),
                           PrintValue());
    
    TEST_P(SortTest, BitonicSort) {
      auto v = generate_unique_ptrs_vec(GetParam());
      auto begin = v.data();

      gcsort::smallsort::bitonic_sort_int64_t((int64_t *) begin, GetParam());

      EXPECT_THAT(v, WhenSorted(ElementsAreArray(v)));
    }

}